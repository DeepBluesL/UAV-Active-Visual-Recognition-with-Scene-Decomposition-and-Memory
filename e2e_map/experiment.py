"""Train every ResNet50 backbone layer jointly with the MAP/GRU TD objective.

Frozen recognition scores and the original environment remain authoritative.
Resume checkpoints include replay, environments, optimizer, scaler, and RNG.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from map_baseline.environment import Dataset, Protocol, TRAIN, LEGAL, RANGES, save_manifest
from map_baseline.features import sha256
from map_baseline.model import masked_argmax
from map_baseline.checkpoints import SnapshotSchedule
from ibe_map.experiment import PrefixReplay, validate_cache_for_variant
from ibe_map.multiseed import write_json
from .model import E2EModel, E2ERunner
from .images import RGBImageStore

ROOT = Path(__file__).resolve().parents[1]


def td_loss(runner, model, target, replay, sample, gamma):
    histories, lengths = replay.histories[sample], replay.lengths[sample]
    current = runner.current_q(model, histories, lengths)
    actions = torch.as_tensor(replay.actions[sample], device=runner.device)
    estimate = current.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        expected = torch.as_tensor(replay.rewards[sample], device=runner.device).clone()
        nonterminal = np.flatnonzero(~replay.done[sample])
        # A terminal transition has no bootstrap. In particular, the final
        # sixth observation need not pass through either policy CNN. This
        # leaves the sampled transitions and mean-loss denominator unchanged.
        if len(nonterminal):
            prefix, size = histories[nonterminal], lengths[nonterminal]
            next_online, ids = runner.next_q(model, prefix, size)
            next_action = masked_argmax(next_online, runner.legal[ids % 605])
            next_target, _ = runner.next_q(target, prefix, size)
            bootstrap = next_target.gather(1, next_action[:, None]).squeeze(1)
            expected[torch.as_tensor(nonterminal, device=runner.device)] += gamma * bootstrap
    return nn.functional.smooth_l1_loss(estimate, expected)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    scenes = sorted(args.scenes or TRAIN)
    data = Dataset(args.cache, scenes)
    validate_cache_for_variant(data, args.variant)
    weights = args.weights
    if sha256(weights) != data.observation_signature['pretrained_sha256']:
        raise ValueError('Backbone initialization differs from frozen baseline')
    images = RGBImageStore(data, args.data, args.decomposed, args.image_cache, args.variant,
                           observation_cache=args.cache)
    runner = E2ERunner(data, images, args.device)
    model = E2EModel(args.variant, weights, microbatch=args.microbatch,
                     checkpoint_encoder=not args.no_activation_checkpoint, amp=runner.device.type == 'cuda')
    model.to(runner.device, memory_format=torch.channels_last)
    target = copy.deepcopy(model).eval().requires_grad_(False)
    for network in (model, target):
        if network.recurrent:
            network.head.gru.flatten_parameters()
    runner.enable_target_cache(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=runner.device.type == 'cuda', init_scale=128.)
    protocol = Protocol()
    replay = PrefixReplay(args.replay_capacity, protocol.max_decisions)
    allowed = data.indices(scenes)
    current = rng.choice(allowed, size=args.environments)
    histories = np.full((args.environments, protocol.max_decisions + 1), -1, np.int64)
    histories[:, 0] = current
    decision = np.zeros(args.environments, np.int64)
    env_rows = np.arange(args.environments)
    ep_return, ep_distance = np.zeros(args.environments), np.zeros(args.environments)
    completed = transitions = updates = attempts = skipped = 0
    history, recent = [], []
    elapsed_before = 0.
    next_log = args.log_every
    args.out.mkdir(parents=True, exist_ok=True)
    save_manifest(args.out / 'splits.json')
    config = {'variant': args.variant, 'phase': 'full' if scenes == sorted(TRAIN) else 'smoke',
        'seed': args.seed, 'requested_episodes': args.episodes, 'training_scenes': scenes,
        'validation_scenes': [], 'protocol': protocol.as_dict(), 'learning_rate': 1e-4,
        'optimizer': 'Adam', 'batch_size': args.batch_size, 'environments': args.environments,
        'replay_capacity': args.replay_capacity, 'learning_starts': args.learning_starts,
        'updates_per_vector_step': 1, 'target_update_every': args.target_every,
        'epsilon': {'start': 1., 'final': .05, 'linear_decay_fraction': .7},
        'double_dqn': True, 'loss': 'smooth_l1', 'gradient_clip_norm': 10.,
        'encoder': 'end-to-end ImageNet ResNet50 V1; all backbone convolution and BN affine parameters trainable',
        'batchnorm': 'fixed pretrained running statistics; affine weight and bias trainable',
        'precision': 'FP32 weights and policy heads; CUDA bfloat16 encoder autocast; FP32 feature boundary; GradScaler initial128',
        'normalization': 'same PIL bilinear224 ImageNet mean/std; FP32 feature boundary matched to BF16 frozen controls',
        'microbatch': args.microbatch, 'activation_checkpoint': not args.no_activation_checkpoint,
        'recurrent': args.variant in ('memo_map', 'ibe_map'),
        'replay': 'uniform transition IDs and complete causal prefixes; no stale online image features or hidden states',
        'checkpoint_selection': 'final requested training budget', 'dataset_fingerprint': data.fingerprint,
        'cache_path': str(args.cache.resolve()), 'observation_signature': data.observation_signature,
        'pretrained_sha256': sha256(weights), 'device': str(runner.device),
        'rgb_cache_fingerprint': images.fingerprint,
        'torch_version': str(torch.__version__),
        'gpu': torch.cuda.get_device_name() if runner.device.type == 'cuda' else None,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    initial = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
    schedule = SnapshotSchedule(args.out, args.save_every, args.episodes)

    def checkpoint(path):
        atomic_save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'config': config, 'completed_episodes': completed, 'transitions': transitions, 'updates': updates}, path)

    def save_resume():
        state = {'model': model.state_dict(), 'target': target.state_dict(),
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'config': config,
            'completed': completed, 'transitions': transitions, 'updates': updates, 'attempts': attempts,
            'skipped': skipped, 'history': history, 'recent': recent, 'rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'elapsed': elapsed_before + time.perf_counter() - started, 'next_log': next_log,
            'environment': {k: torch.from_numpy(v) for k, v in dict(current=current, histories=histories,
                decision=decision, ep_return=ep_return, ep_distance=ep_distance).items()},
            'replay': {k: torch.from_numpy(getattr(replay, k)) for k in ('histories', 'lengths', 'actions', 'rewards', 'done')},
            'replay_position': replay.pos, 'replay_size': replay.size,
            'target_feature_cache': runner.target_cache_state()}
        atomic_save(state, args.out / 'resume.pt')

    resume_path = args.out / 'resume.pt'
    if args.resume and resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError('Resume configuration changed')
        model.load_state_dict(saved['model']); target.load_state_dict(saved['target'])
        for network in (model, target):
            if network.recurrent:
                network.head.gru.flatten_parameters()
        runner.load_target_cache_state(saved.get('target_feature_cache'))
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        completed, transitions, updates = saved['completed'], saved['transitions'], saved['updates']
        attempts, skipped = saved['attempts'], saved['skipped']
        history, recent, elapsed_before, next_log = saved['history'], saved['recent'], saved['elapsed'], saved['next_log']
        env = {k: v.numpy().copy() for k, v in saved['environment'].items()}
        current, histories, decision = env['current'], env['histories'], env['decision']
        ep_return, ep_distance = env['ep_return'], env['ep_distance']
        for key, value in saved['replay'].items():
            setattr(replay, key, value.numpy().copy())
        replay.pos, replay.size = saved['replay_position'], saved['replay_size']
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        schedule.next_index = sum(point <= completed for point in schedule.points)
        del saved
        print(json.dumps({'resumed_episodes': completed, 'updates': updates}), flush=True)
        if completed >= args.episodes:
            return json.loads((args.out / 'training.json').read_text(encoding='utf8'))
    else:
        if (args.out / 'training.json').exists():
            raise ValueError('Existing training output; use --resume or a new directory')
        schedule.capture(0, checkpoint)
    started = time.perf_counter()
    start_attempts = attempts
    loss_value = None
    while completed < args.episodes:
        model.eval()
        epsilon = max(.05, 1. - .95 * completed / max(1, args.episodes * .7))
        a = runner.choose(model, histories, decision + 1)
        explore = rng.random(args.environments) < epsilon
        scores = rng.random((args.environments, 25)); scores[~LEGAL[current % 605]] = -1
        a[explore] = scores.argmax(1)[explore]
        following, reward, done, _ = data.transition(current, a, decision, protocol)
        histories[env_rows, decision + 1] = following
        replay.add(histories, decision + 2, a, reward, done)
        ep_return += protocol.gamma ** decision * reward
        ep_distance += RANGES[a] * 10
        decision += 1; transitions += len(a)
        for i in np.flatnonzero(done):
            recent.append((float(ep_return[i]), float(ep_distance[i]), bool(data.correct[following[i]])))
        recent = recent[-5000:]
        completed += int(done.sum())
        current = following.copy(); current[done] = rng.choice(allowed, size=int(done.sum()))
        histories[done] = -1; histories[done, 0] = current[done]
        decision[done] = 0; ep_return[done] = 0; ep_distance[done] = 0
        if replay.size >= args.learning_starts:
            model.train()
            sample = rng.integers(replay.size, size=args.batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = td_loss(runner, model, target, replay, sample, protocol.gamma)
            if not torch.isfinite(loss):
                atomic_save({'state_dict': model.state_dict(), 'target_state_dict': target.state_dict(),
                    'config': config, 'completed_episodes': completed, 'updates': updates,
                    'histories': torch.from_numpy(replay.histories[sample]),
                    'lengths': torch.from_numpy(replay.lengths[sample])}, args.out / 'nonfinite_failure.pt')
                raise RuntimeError(f'Non-finite TD loss at episode {completed}, update {updates}; diagnostic saved')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            attempts += 1
            if scaler.get_scale() < old_scale:
                skipped += 1
            else:
                updates += 1
                if updates % args.target_every == 0:
                    target.load_state_dict(model.state_dict())
                    if hasattr(runner, 'invalidate_target'):
                        runner.invalidate_target()
            loss_value = float(loss.detach())
        schedule.capture(completed, checkpoint)
        stopping = args.stop_after_updates and attempts - start_attempts >= args.stop_after_updates
        if completed >= next_log or completed >= args.episodes or stopping:
            row = {'episodes': completed, 'transitions': transitions, 'updates': updates,
                'optimizer_attempts': attempts, 'amp_skipped_updates': skipped, 'epsilon': epsilon,
                'loss': loss_value, 'train_recent_return': float(np.mean([x[0] for x in recent])),
                'train_recent_accuracy': float(np.mean([x[2] for x in recent])),
                'elapsed_seconds': elapsed_before + time.perf_counter() - started}
            changes = {}
            for name in ('conv1.weight', 'layer1.0.conv1.weight', 'layer2.0.conv1.weight',
                         'layer3.0.conv1.weight', 'layer4.0.conv1.weight', 'bn1.weight'):
                value = model.encoder.state_dict()[name].detach().cpu()
                changes[name] = float((value - initial[name]).norm())
            bn_fixed = all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.encoder.state_dict().items()
                           if 'running_' in k or 'num_batches_tracked' in k)
            if not bn_fixed:
                raise RuntimeError('BatchNorm running statistics changed')
            history.append(row)
            checkpoint(args.out / 'last.pt')
            summary = {'config': config, 'completed_episodes': completed, 'transitions': transitions,
                'updates': updates, 'optimizer_attempts': attempts, 'amp_skipped_updates': skipped,
                'elapsed_seconds': row['elapsed_seconds'], 'history': history,
                'backbone_parameter_l2_change': changes, 'batchnorm_running_stats_unchanged': bn_fixed,
                'complete': completed >= args.episodes}
            write_json(args.out / 'training.json', summary)
            while next_log <= completed:
                next_log += args.log_every
            save_resume()
            print(json.dumps({**row, 'backbone_l2_change': changes, 'complete': summary['complete']}), flush=True)
            if stopping:
                break
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('map', 'memo_map', 'sam_map', 'ibe_map'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--decomposed', type=Path, default=ROOT / 'output/ibe_map/decomposed')
    p.add_argument('--image-cache', type=Path, default=ROOT / 'output/e2e_images')
    p.add_argument('--weights', type=Path, default=ROOT / 'output/map/cache/weights/resnet50-0676ba61.pth')
    p.add_argument('--scenes', type=int, nargs='+')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--episodes', type=int, default=300000)
    p.add_argument('--environments', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--replay-capacity', type=int, default=100000)
    p.add_argument('--learning-starts', type=int, default=2048)
    p.add_argument('--target-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=5000)
    p.add_argument('--save-every', type=int, default=30000)
    p.add_argument('--microbatch', type=int, default=32)
    p.add_argument('--no-activation-checkpoint', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-updates', type=int, default=0, help='Smoke/throughput test only; saves resumable incomplete run')
    p.add_argument('--device', default='cuda')
    return p


if __name__ == '__main__':
    p = parser(); args = p.parse_args()
    if args.cache is None:
        args.cache = ROOT / ('output/ibe_map/cache' if args.variant in ('sam_map', 'ibe_map') else 'output/map/cache')
    if min(args.episodes, args.environments, args.batch_size, args.target_every, args.log_every, args.microbatch) < 1:
        p.error('Training sizes and intervals must be positive')
    if args.learning_starts < args.batch_size or args.replay_capacity < max(args.learning_starts, args.environments):
        p.error('Invalid replay size / warmup')
    train(args)
