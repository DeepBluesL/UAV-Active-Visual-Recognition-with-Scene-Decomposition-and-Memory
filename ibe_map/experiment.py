"""Train causal recurrent/SAM policies; evaluate all methods on matched starts.

The immutable environment and classifier are imported from the MAP baseline.
Replay samples transitions uniformly, with each transition carrying its entire
episode prefix. No stale hidden state is saved or reused after optimizer steps.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from map_baseline.environment import (CLASSES, DEVELOPMENT, GRID, HARD, LEGAL,
    RANGES, SIMPLE, STOP, TRAIN, VALIDATION, Dataset, Protocol, heuristic_actions, save_manifest)
from map_baseline.experiment import aggregate, make_example
from map_baseline.model import masked_argmax
from map_baseline.checkpoints import SnapshotSchedule
from .model import make_model

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ('map', 'memo_map', 'sam_map', 'ibe_map')
POLICIES = ('stationary', 'sweep', 'random') + LEARNED


def validate_cache_for_variant(data, variant):
    """Reject mislabeled ablations even when a checkpoint matches its cache."""
    if variant not in POLICIES:
        raise ValueError(f'Unknown policy variant: {variant}')
    preprocess = data.observation_signature.get('preprocess')
    if not isinstance(preprocess, dict):
        raise ValueError('Observation cache is missing a preprocessing signature')
    decomposed = 'scene_decomposition' in preprocess
    requires_sam = variant in ('sam_map', 'ibe_map')
    if requires_sam:
        marker = preprocess.get('scene_decomposition')
        if not isinstance(marker, dict) or not marker:
            raise ValueError(f'{variant} requires a SAM scene_decomposition cache; raw images cannot be labeled SAM')
    elif decomposed:
        raise ValueError(f'{variant} requires the original raw-image cache, without scene_decomposition')


class SequenceRunner:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.device = torch.device(device if device != 'cuda' or torch.cuda.is_available() else 'cpu')
        self.features = torch.as_tensor(data.features, device=self.device)
        self.boxes = torch.as_tensor(data.boxes, device=self.device)
        self.legal = torch.as_tensor(LEGAL, device=self.device)

    def sequence_q(self, model, histories, lengths):
        """Return causal Q at every prefix step, ignoring right-padding outputs."""
        longest = int(np.max(lengths))
        # The padding index is replaced only for tensor gathering. A future
        # padding frame cannot alter an earlier unidirectional GRU output.
        ids = torch.as_tensor(np.maximum(histories[:, :longest], 0), dtype=torch.long, device=self.device)
        q, _ = model.forward_sequence(self.features[ids], self.boxes[ids])
        return q, ids

    def transition_q(self, model, histories, lengths):
        """Current/next Q, both reconstructed independently from episode start."""
        rows = torch.arange(len(histories), device=self.device)
        previous = torch.as_tensor(lengths - 2, dtype=torch.long, device=self.device)
        following = previous + 1
        if model.recurrent:
            q, ids = self.sequence_q(model, histories, lengths)
            return q[rows, previous], q[rows, following], ids[rows, following]
        ids = torch.as_tensor(histories, dtype=torch.long, device=self.device)
        current_ids, next_ids = ids[rows, previous], ids[rows, following]
        current_q = model(self.features[current_ids], self.boxes[current_ids])
        next_q = model(self.features[next_ids], self.boxes[next_ids])
        return current_q, next_q, next_ids

    @torch.inference_mode()
    def choose(self, model, histories, lengths, allow_stop=True):
        chosen = []
        for start in range(0, len(histories), 512):
            h = histories[start:start + 512]
            n = lengths[start:start + 512]
            rows = torch.arange(len(h), device=self.device)
            positions = torch.as_tensor(n - 1, dtype=torch.long, device=self.device)
            if model.recurrent:
                all_q, all_ids = self.sequence_q(model, h, n)
                q, ids = all_q[rows, positions], all_ids[rows, positions]
            else:
                ids = torch.as_tensor(h[np.arange(len(h)), n - 1], device=self.device)
                q = model(self.features[ids], self.boxes[ids])
            valid = self.legal[ids % 605].clone()
            if not allow_stop:
                valid[:, STOP] = False
            chosen.append(masked_argmax(q, valid).cpu().numpy())
        return np.concatenate(chosen)


class PrefixReplay:
    """Uniform transition ring, storing the whole causal episode prefix.

    This retains the original 100,000-transition capacity, sampling weights,
    and batch size. Prefix copies are independent of later environment resets.
    Each sampled transition contributes exactly one loss; padding contributes
    none. Hidden states are never cached in replay.
    """
    def __init__(self, capacity, horizon=5):
        self.capacity, self.horizon = capacity, horizon
        self.pos = self.size = 0
        self.histories = np.full((capacity, horizon + 1), -1, np.int64)
        self.lengths = np.empty(capacity, np.int64)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, bool)

    def add(self, histories, lengths, actions, rewards, done):
        if np.any(lengths < 2) or np.any(lengths > self.horizon + 1):
            raise ValueError('Replay transition must contain current and next observations')
        slots = (np.arange(len(histories)) + self.pos) % self.capacity
        for name, value in (('histories', histories), ('lengths', lengths), ('actions', actions),
                            ('rewards', rewards), ('done', done)):
            getattr(self, name)[slots] = value
        self.pos = (self.pos + len(histories)) % self.capacity
        self.size = min(self.capacity, self.size + len(histories))


def td_loss(runner, model, target, replay, sample, gamma):
    histories, lengths = replay.histories[sample], replay.lengths[sample]
    current_q, next_online, next_ids = runner.transition_q(model, histories, lengths)
    actions = torch.as_tensor(replay.actions[sample], device=runner.device)
    estimate = current_q.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_action = masked_argmax(next_online.detach(), runner.legal[next_ids % 605])
        _, next_target, _ = runner.transition_q(target, histories, lengths)
        bootstrap = next_target.gather(1, next_action[:, None]).squeeze(1)
        reward = torch.as_tensor(replay.rewards[sample], device=runner.device)
        terminal = torch.as_tensor(replay.done[sample], device=runner.device)
        expected = reward + gamma * (~terminal) * bootstrap
    return nn.functional.smooth_l1_loss(estimate, expected)


def rollout(runner, model, starts, policy, protocol, seed=42, mode='native', fixed_moves=3):
    data = runner.data
    n = len(starts)
    current = np.asarray(starts).copy()
    initial = current.copy()
    active = np.ones(n, bool)
    returns, distance = np.zeros(n, np.float64), np.zeros(n, np.float64)
    decisions = np.zeros(n, np.int16)
    actions = np.full((n, protocol.max_decisions), -1, np.int16)
    observations = np.full((n, protocol.max_decisions + 1), -1, np.int64)
    rewards = np.zeros((n, protocol.max_decisions), np.float32)
    observations[:, 0] = current
    rng = np.random.default_rng(seed)
    visited = [{int(idx % 605)} for idx in starts]
    for t in range(protocol.max_decisions):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        idx = current[rows]
        if policy in LEARNED:
            if mode == 'fixed_moves' and t >= fixed_moves:
                a = np.full(len(rows), STOP, np.int64)
            else:
                a = runner.choose(model, observations[rows], decisions[rows] + 1, allow_stop=mode != 'fixed_moves')
        else:
            a = heuristic_actions(data, idx, t, policy, rng, [visited[i] for i in rows], mode,
                                  fixed_moves, stop_threshold=protocol.heuristic_stop_threshold)
        nxt, reward, done, _ = data.transition(idx, a, t, protocol)
        actions[rows, t], rewards[rows, t] = a, reward
        observations[rows, t + 1] = nxt
        returns[rows] += protocol.gamma ** t * reward
        distance[rows] += RANGES[a] * 10
        decisions[rows] += 1
        current[rows] = nxt
        active[rows[done]] = False
        for row, pos in zip(rows, nxt):
            visited[row].add(int(pos % 605))
    if active.any():
        raise AssertionError('Episode horizon failed to terminate all environments')
    success = data.visible[current] & (data.true_probability[current] > protocol.recognition_threshold)
    if protocol.success_mode == 'top1_and_true_probability':
        success &= data.correct[current]
    return {'starts': initial, 'final': current, 'returns': returns, 'distance': distance,
            'decisions': decisions, 'actions': actions, 'observations': observations, 'rewards': rewards,
            'correct': data.correct[current], 'initial_correct': data.correct[initial], 'success': success}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    protocol = Protocol(success_mode=args.success_mode)
    scenes = args.scenes or (DEVELOPMENT if args.phase == 'development' else TRAIN)
    validate = VALIDATION if args.phase == 'development' and not args.scenes else []
    data = Dataset(args.cache, sorted(set(scenes + validate)))
    validate_cache_for_variant(data, args.variant)
    runner = SequenceRunner(data, args.device)
    model, target = make_model(args.variant).to(runner.device), make_model(args.variant).to(runner.device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    replay = PrefixReplay(args.replay_capacity, protocol.max_decisions)
    allowed = data.indices(scenes)
    current = rng.choice(allowed, size=args.environments)
    histories = np.full((args.environments, protocol.max_decisions + 1), -1, np.int64)
    histories[:, 0] = current
    decision = np.zeros(args.environments, np.int64)
    env_rows = np.arange(args.environments)
    ep_return, ep_distance = np.zeros(args.environments), np.zeros(args.environments)
    completed = transitions = updates = 0
    history, recent = [], []
    started, next_log, best = time.perf_counter(), args.log_every, -float('inf')
    args.out.mkdir(parents=True, exist_ok=True)
    snapshots = SnapshotSchedule(args.out, getattr(args, 'save_every', 0), args.episodes)
    save_manifest(args.out / 'splits.json')
    config = {'variant': args.variant, 'phase': args.phase, 'seed': args.seed,
        'requested_episodes': args.episodes, 'training_scenes': scenes, 'validation_scenes': validate,
        'protocol': protocol.as_dict(), 'learning_rate': 1e-4, 'optimizer': 'Adam',
        'batch_size': args.batch_size, 'environments': args.environments, 'replay_capacity': args.replay_capacity,
        'learning_starts': args.learning_starts, 'updates_per_vector_step': args.updates_per_step,
        'target_update_every': args.target_every, 'epsilon': {'start': 1., 'final': .05, 'linear_decay_fraction': .7},
        'double_dqn': True, 'loss': 'smooth_l1', 'gradient_clip_norm': 10.,
        'encoder': 'frozen ImageNet ResNet50 V1',
        'network': 'image512+bbox32=g544; GRU hidden544; state512; V1+direction7+range4' if model.recurrent
                   else 'image512+bbox32=g544; state512; V1+direction7+range4 (same MAP architecture)',
        'recurrent': model.recurrent,
        'replay': 'uniform transitions, each with full episode prefix; zero hidden and independent online/target unroll',
        'acting_hidden': 'recomputed from full causal prefix after each optimizer update; reset at episode start',
        'padding_loss': 'one current-transition loss per sampled prefix; no loss on padding',
        'normalization': 'per-observation LayerNorm on 2048 feature; bbox /256',
        'checkpoint_selection': 'best validation return' if validate else 'final requested training budget',
        'dataset_fingerprint': data.fingerprint, 'cache_path': str(args.cache.resolve()),
        'observation_signature': data.observation_signature, 'device': str(runner.device),
        'torch_version': str(torch.__version__),
        'gpu': torch.cuda.get_device_name() if runner.device.type == 'cuda' else None}

    def checkpoint(path):
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'config': config, 'completed_episodes': completed, 'transitions': transitions,
                    'updates': updates}, path)

    snapshots.capture(completed, checkpoint)

    while completed < args.episodes:
        model.eval()
        epsilon = max(.05, 1. - .95 * completed / max(1, args.episodes * .7))
        a = runner.choose(model, histories, decision + 1)
        explore = rng.random(args.environments) < epsilon
        random_scores = rng.random((args.environments, 25))
        random_scores[~LEGAL[current % 605]] = -1
        a[explore] = random_scores.argmax(1)[explore]
        following, reward, done, _ = data.transition(current, a, decision, protocol)
        histories[env_rows, decision + 1] = following
        replay.add(histories, decision + 2, a, reward, done)
        ep_return += protocol.gamma ** decision * reward
        ep_distance += RANGES[a] * 10
        decision += 1
        transitions += len(a)
        for i in np.flatnonzero(done):
            recent.append((float(ep_return[i]), float(ep_distance[i]), bool(data.correct[following[i]])))
        if len(recent) > 5000:
            recent = recent[-5000:]
        completed += int(done.sum())
        current = following.copy()
        current[done] = rng.choice(allowed, size=int(done.sum()))
        histories[done] = -1
        histories[done, 0] = current[done]
        decision[done] = 0
        ep_return[done], ep_distance[done] = 0, 0
        last_loss = None
        if replay.size >= args.learning_starts:
            model.train()
            for _ in range(args.updates_per_step):
                sample = rng.integers(replay.size, size=args.batch_size)
                loss = td_loss(runner, model, target, replay, sample, protocol.gamma)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                updates += 1
                last_loss = float(loss.detach())
                if updates % args.target_every == 0:
                    target.load_state_dict(model.state_dict())
        snapshots.capture(completed, checkpoint)
        if completed >= next_log or completed >= args.episodes:
            model.eval()
            row = {'episodes': completed, 'transitions': transitions, 'updates': updates, 'epsilon': epsilon,
                'loss': last_loss, 'train_recent_return': float(np.mean([x[0] for x in recent])),
                'train_recent_accuracy': float(np.mean([x[2] for x in recent])),
                'elapsed_seconds': time.perf_counter() - started}
            if validate:
                validation = aggregate(rollout(runner, model, data.indices(validate), args.variant, protocol, args.seed))
                row.update({f'validation_{k}': v for k, v in validation.items()})
                if validation['return_mean'] > best:
                    best = validation['return_mean']
                    checkpoint(args.out / 'best.pt')
            history.append(row)
            checkpoint(args.out / 'last.pt')
            summary = {'config': config, 'completed_episodes': completed, 'transitions': transitions,
                       'updates': updates, 'elapsed_seconds': time.perf_counter() - started,
                       'history': history, 'best_validation_return': best if validate else None}
            (args.out / 'training.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
            print(json.dumps(row), flush=True)
            next_log += args.log_every
    return summary


def verify_shared_observations(raw, sam):
    if raw.scenes != sam.scenes:
        raise ValueError('Raw/SAM caches contain different scene identities')
    for key in ('classifier_sha256', 'dataset_path', 'pretrained_sha256'):
        if raw.observation_signature[key] != sam.observation_signature[key]:
            raise ValueError(f'Raw/SAM cache {key} mismatch')
    for key in ('boxes', 'probs', 'visible', 'names'):
        if not np.array_equal(getattr(raw, key), getattr(sam, key)):
            raise ValueError(f'Raw/SAM cache {key} differs; recognition and observations must stay matched')


def evaluate(args):
    from demo.render_results import render
    torch.set_num_threads(4)
    split_scenes = {'train': TRAIN, 'simple': SIMPLE, 'hard': HARD}
    selected = {key: split_scenes[key] for key in args.splits}
    scenes = sorted({s for group in selected.values() for s in group})
    raw = Dataset(args.raw_cache, scenes)
    validate_cache_for_variant(raw, 'map')
    sam = Dataset(args.sam_cache, scenes) if any(p in args.policies for p in ('sam_map', 'ibe_map')) else None
    if sam is not None:
        validate_cache_for_variant(sam, 'ibe_map')
        verify_shared_observations(raw, sam)
    if args.data.resolve() != Path(raw.observation_signature['dataset_path']).resolve():
        raise ValueError('Visualization dataset differs from observation-cache source')
    raw_runner = SequenceRunner(raw, args.device)
    sam_runner = SequenceRunner(sam, args.device) if sam is not None else None
    runners = {p: (sam_runner if p in ('sam_map', 'ibe_map') else raw_runner) for p in args.policies}
    models, checkpoints, training, checkpoint_hashes = {}, {}, {}, {}
    protocol = None
    for policy in args.policies:
        validate_cache_for_variant(runners[policy].data, policy)
        if policy not in LEARNED:
            continue
        path = getattr(args, f'{policy}_checkpoint')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        expected_variant = checkpoint['config'].get('variant', 'map')
        if expected_variant != policy:
            raise ValueError(f'Expected {policy} checkpoint, got {expected_variant}: {path}')
        data = runners[policy].data
        if checkpoint['config'].get('observation_signature') != data.observation_signature:
            raise ValueError(f'{policy} checkpoint/cache observation signature mismatch')
        current_protocol = Protocol(**checkpoint['config']['protocol'])
        if protocol is not None and protocol != current_protocol:
            raise ValueError('All policy checkpoints must share the same environment/reward protocol')
        protocol = current_protocol
        model = make_model(policy).to(runners[policy].device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        models[policy], checkpoints[policy] = model, checkpoint
        training_path = path.parent / 'training.json'
        training[policy] = json.loads(training_path.read_text()) if training_path.exists() else checkpoint['config']
        checkpoint_hashes[policy] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = protocol or Protocol()
    args.out.mkdir(parents=True, exist_ok=True)
    save_manifest(args.out / 'splits.json')
    metrics, examples = [], []
    fields = ['split', 'mode', 'policy', 'scene', 'start_x', 'start_y', 'start_z', 'initial_correct',
              'final_correct', 'return', 'distance_m', 'decisions', 'stop_reason', 'final_confidence', 'success']
    with (args.out / 'evaluation.csv').open('w', newline='', encoding='utf8') as f, \
            gzip.open(args.out / 'trajectories.jsonl.gz', 'wt', encoding='utf8') as trace:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, group in selected.items():
            starts = raw.indices(group)
            for mode in (['native', 'fixed_moves'] if split == 'hard' else ['native']):
                policies = [p for p in args.policies if mode == 'native' or p != 'stationary']
                for policy in policies:
                    runner = runners[policy]
                    result = rollout(runner, models.get(policy), starts, policy, protocol, args.seed, mode)
                    summary = {'split': split if mode == 'native' else 'hard_fixed_3_moves',
                               'policy': policy, **aggregate(result)}
                    per_scene = []
                    for scene in group:
                        mask = starts // 605 == raw.scene_to_slot[scene]
                        per_scene.append([float(result['correct'][mask].mean()), float(result['returns'][mask].mean())])
                    values = np.array(per_scene)
                    bootstrap_rng = np.random.default_rng(1729)
                    sampled = values[bootstrap_rng.integers(len(values), size=(2000, len(values)))].mean(1)
                    summary['accuracy_scene_bootstrap_95ci'] = np.quantile(sampled[:, 0], [.025, .975]).tolist()
                    summary['return_scene_bootstrap_95ci'] = np.quantile(sampled[:, 1], [.025, .975]).tolist()
                    metrics.append(summary)
                    print(json.dumps(summary), flush=True)
                    for row, idx in enumerate(starts):
                        scene = raw.scenes[idx // 605]
                        count = int(result['decisions'][row])
                        taken = result['actions'][row, :count].tolist()
                        reason = 'policy_stop' if taken[-1] == STOP else 'max_decisions_after_move'
                        if taken[-1] == STOP and mode == 'fixed_moves':
                            reason = 'fixed_movement_budget'
                        elif taken[-1] == STOP and policy in ('sweep', 'random'):
                            reason = 'confidence_threshold'
                        final = int(result['final'][row])
                        record = {'split': split, 'mode': mode, 'policy': policy, 'scene': scene,
                            'start_x': int(GRID[idx % 605, 0]), 'start_y': int(GRID[idx % 605, 1]),
                            'start_z': int(GRID[idx % 605, 2]), 'initial_correct': bool(result['initial_correct'][row]),
                            'final_correct': bool(result['correct'][row]), 'return': float(result['returns'][row]),
                            'distance_m': float(result['distance'][row]), 'decisions': count, 'stop_reason': reason,
                            'final_confidence': float(raw.confidence[final]), 'success': bool(result['success'][row])}
                        writer.writerow(record)
                        observed = result['observations'][row, :count + 1]
                        trace.write(json.dumps({**record, 'positions': GRID[observed % 605].tolist(), 'actions': taken,
                            'rewards': result['rewards'][row, :count].tolist(),
                            'predictions': [CLASSES[int(raw.predicted[i])] if raw.visible[i] else 'Unknown' for i in observed],
                            'probabilities': raw.probs[observed].tolist()}, separators=(',', ':')) + '\n')
                    if split == 'hard' and mode == 'native' and policy in LEARNED:
                        for rows, note in [
                            (np.flatnonzero(~result['initial_correct'] & result['correct']),
                             'Post-selected wrong-to-correct illustration; aggregate metrics include all starting views.'),
                            (np.flatnonzero(result['initial_correct'] & ~result['correct']),
                             'Post-selected failure: a correct initial view became incorrect.')]:
                            if len(rows):
                                examples.append(make_example(raw, result, int(rows[0]), policy, args.data, note))
    native = {(m['split'], m['policy']): m for m in metrics}
    if 'train' in selected and 'hard' in selected:
        for policy in args.policies:
            native[('hard', policy)]['generalization_gap_return'] = (
                native[('train', policy)]['return_mean'] - native[('hard', policy)]['return_mean'])
    result = {'protocol': {**protocol.as_dict(),
        'initial_positions': 'all 605 viewpoints per scene, identical across policies',
        'action_space': 'six directions x ranges1..4 plus STOP; 10m per unit; identical boundary masks',
        'classification': 'identical frozen original-image ResNet18 outputs in raw and SAM caches',
        'native_stopping': 'learned policies predict STOP; sweep/random retain original 0.90 confidence threshold',
        'fixed_moves_diagnostic': 'exactly three movements then STOP; controls count, not traveled meters',
        'decision_timing': 'move then recognize/reward; fifth decision may move; at most six observations',
        'evaluation_seed': args.seed, 'checkpoint_sha256': checkpoint_hashes,
        'raw_cache_fingerprint': raw.fingerprint, 'sam_cache_fingerprint': sam.fingerprint if sam else None,
        'raw_observation_signature': raw.observation_signature,
        'sam_observation_signature': sam.observation_signature if sam else None,
        'gru': 'g544, one-layer hidden544, s512; zero state per episode; causal prefix replay'},
        'training': training, 'metrics': metrics, 'examples': examples, 'classes': CLASSES,
        'limitations': [
            'Reimplementation using UEVAVD v1 descriptions; no released author policy code or checkpoints.',
            'Frozen ImageNet ResNet50 and the existing ResNet18 classifier are retained for controlled comparison.',
            'GRU dimensions follow the paper; unspecified training, replay and dueling composition choices are documented.',
            'Official pretrained SAM ViT-B is used without UEVAVD SAM fine-tuning: the released dataset supplies bounding boxes, not pixel-level target masks, and author fine-tuned SAM weights were not provided. This differs from the paper.',
            'One policy training seed and one random-policy evaluation seed; scene-bootstrap intervals are not seed uncertainty.',
            'Ground-truth bounding boxes replace online tracking; no physical collision or flight dynamics simulation.',
            'Fixed-three-move diagnostics control action count, not exact path length.',
            'Example sequences are post-selected illustrations and show the original images used by the fixed classifier.'
        ], 'sources': [
            {'title': 'UEVAVD arXiv v1', 'url': 'https://arxiv.org/html/2411.04348v1'},
            {'title': 'Author dataset repository', 'url': 'https://github.com/Leo000ooo/UEVAVD_dataset'},
            {'title': 'Meta official Segment Anything', 'url': 'https://github.com/facebookresearch/segment-anything'}]}
    result_path = args.out / 'results.json'
    result_path.write_text(json.dumps(result, indent=2), encoding='utf8')
    if examples:
        sequence = {'method': 'MAP / Memo-MAP / SAM-MAP / IBE-MAP comparison', 'limitations': result['limitations'],
                    'classes': CLASSES, 'metrics': native.get(('hard', 'ibe_map'), {}), 'episodes': examples}
        (args.out / 'examples.json').write_text(json.dumps(sequence, indent=2), encoding='utf8')
        render(args.out / 'examples.json', args.out / 'examples')
    if not args.no_report:
        subprocess.run([sys.executable, '-c',
            'import sys; from ibe_map.report import render_report; render_report(sys.argv[1],sys.argv[2])',
            str(result_path), str(args.out)], check=True, cwd=ROOT)
    print(f'Results: {result_path.resolve()}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--variant', choices=['memo_map', 'sam_map', 'ibe_map'], required=True)
    training.add_argument('--phase', choices=['development', 'full'], default='development')
    training.add_argument('--cache', type=Path)
    training.add_argument('--scenes', nargs='+', type=int)
    training.add_argument('--episodes', type=int, default=30000)
    training.add_argument('--environments', type=int, default=64)
    training.add_argument('--batch-size', type=int, default=256)
    training.add_argument('--replay-capacity', type=int, default=100000)
    training.add_argument('--learning-starts', type=int, default=2048)
    training.add_argument('--updates-per-step', type=int, default=1)
    training.add_argument('--target-every', type=int, default=500)
    training.add_argument('--log-every', type=int, default=5000)
    training.add_argument('--save-every', type=int, default=0,
                          help='Save initial/intermediate/final snapshots; 0 disables diagnostic snapshots')
    training.add_argument('--success-mode', choices=['top1_and_true_probability', 'true_probability'], default='top1_and_true_probability')
    evaluating = sub.add_parser('evaluate')
    evaluating.add_argument('--raw-cache', type=Path, default=ROOT / 'output/map/cache')
    evaluating.add_argument('--sam-cache', type=Path, default=ROOT / 'output/ibe_map/cache')
    for variant in LEARNED:
        default = ROOT / ('output/map/full/last.pt' if variant == 'map' else f'output/ibe_map/{variant}/full/last.pt')
        evaluating.add_argument(f'--{variant.replace("_", "-")}-checkpoint', dest=f'{variant}_checkpoint', type=Path, default=default)
    evaluating.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    evaluating.add_argument('--data', type=Path, default=ROOT / 'dataset')
    evaluating.add_argument('--splits', nargs='+', choices=['train', 'simple', 'hard'], default=['train', 'simple', 'hard'])
    evaluating.add_argument('--no-report', action='store_true', help='Write full results/trajectories/examples without HTML summary rendering')
    for command in (training, evaluating):
        command.add_argument('--out', type=Path)
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'train':
        if args.cache is None:
            args.cache = ROOT / ('output/map/cache' if args.variant == 'memo_map' else 'output/ibe_map/cache')
        if args.out is None:
            args.out = ROOT / 'output/ibe_map' / args.variant / args.phase
        if (args.episodes < 1 or args.environments < 1 or args.learning_starts < args.batch_size
                or args.replay_capacity < args.learning_starts or args.replay_capacity < args.environments
                or args.target_every < 1 or args.log_every < 1):
            parser.error('Invalid training budget, environment count, replay sizes or target/log frequency')
        train(args)
    else:
        if len(set(args.policies)) != len(args.policies):
            parser.error('Duplicate policy names')
        if args.out is None:
            args.out = ROOT / 'output/ibe_map'
        evaluate(args)


if __name__ == '__main__':
    main()
