"""Evaluate one fixed end-to-end checkpoint on the existing complete grid.

Feature reuse is restricted to evaluation after the encoder has stopped changing.
Classification, actions, trajectory schema, and independent verifier are shared.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from map_baseline.environment import CLASSES, GRID, HARD, SIMPLE, TRAIN, STOP, Dataset, Protocol, save_manifest
from map_baseline.experiment import aggregate, make_example
from ibe_map.experiment import SequenceRunner, rollout, validate_cache_for_variant, verify_shared_observations, LEARNED
from .images import RGBImageStore
from .model import E2EModel, E2ERunner
ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate(args):
    from demo.render_results import render
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    policy = config['variant']
    if checkpoint['completed_episodes'] < config['requested_episodes'] or config['phase'] != 'full':
        raise ValueError('Only completed final full-TRAIN checkpoints can be formally evaluated')
    args.seed = config['seed']
    args.policies = [policy]
    selected = {name: group for name, group in {'train': TRAIN, 'simple': SIMPLE, 'hard': HARD}.items()
                if name in args.splits}
    scenes = sorted(set(scene for group in selected.values() for scene in group))
    raw = Dataset(args.raw_cache, scenes)
    sam = Dataset(args.sam_cache, scenes) if policy in ('sam_map', 'ibe_map') else None
    if sam is not None:
        verify_shared_observations(raw, sam)
    data = sam if sam is not None else raw
    validate_cache_for_variant(data, policy)
    if config['observation_signature'] != data.observation_signature:
        raise ValueError('Checkpoint and source observation identities differ')
    protocol = Protocol(**config['protocol'])
    images = RGBImageStore(data, args.data, args.decomposed, args.image_cache, policy,
                           observation_cache=args.sam_cache if sam is not None else args.raw_cache)
    direct = E2ERunner(data, images, args.device)
    model = E2EModel(policy, args.weights, microbatch=args.microbatch,
                     checkpoint_encoder=False, amp=direct.device.type == 'cuda').to(direct.device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    data.features = np.empty((len(data.names), 2048), dtype=np.float32)
    for start in range(0, len(data.names), 128):
        ids = np.arange(start, min(start + 128, len(data.names)))
        data.features[ids] = direct.image_features(model, ids).cpu().numpy()
        if start % (128 * 100) == 0:
            print(f'Final checkpoint feature extraction {start}/{len(data.names)}', flush=True)
    if not np.isfinite(data.features).all():
        raise ValueError('Final encoder produced non-finite evaluation features')
    # Only this immutable final encoder is cached; the policy head is unchanged.
    runner = SequenceRunner(data, args.device)
    runners = {policy: runner}
    models = {policy: model.head}
    checkpoints = {policy: checkpoint}
    training = {policy: json.loads((args.checkpoint.parent / 'training.json').read_text(encoding='utf8'))}
    checkpoint_hashes = {policy: hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
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
        'evaluation_splits': list(selected),
        'raw_cache_fingerprint': raw.fingerprint, 'sam_cache_fingerprint': sam.fingerprint if sam else None,
        'raw_observation_signature': raw.observation_signature,
        'sam_observation_signature': sam.observation_signature if sam else None,
        'gru': 'g544, one-layer hidden544, s512; zero state per episode; causal prefix replay',
        'encoder_training': 'end-to-end all ResNet50 layers, fixed BN running statistics',
        'encoder_precision': 'CUDA bfloat16 autocast; FP32 features and policy heads',
        'evaluation_features': 'computed once from this exact final checkpoint, never used during training',
        'rgb_cache_fingerprint': images.fingerprint},
        'training': training, 'metrics': metrics, 'examples': examples, 'classes': CLASSES,
        'limitations': [
            'Reimplementation using UEVAVD v1 descriptions; no released author policy code or checkpoints.',
            'ResNet50 is jointly trained by TD gradients; recognition still uses the identical frozen ResNet18 classifier.',
            'GRU dimensions follow the paper; unspecified training, replay and dueling composition choices are documented.',
            'Official pretrained SAM ViT-B is used without UEVAVD SAM fine-tuning: the released dataset supplies bounding boxes, not pixel-level target masks, and author fine-tuned SAM weights were not provided. This differs from the paper.',
            'This file records one training seed; the final comparison aggregates five matched seeds separately.',
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
    print(f'Results: {result_path.resolve()}', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--raw-cache', type=Path, default=ROOT / 'output/map/cache')
    p.add_argument('--sam-cache', type=Path, default=ROOT / 'output/ibe_map/cache')
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--decomposed', type=Path, default=ROOT / 'output/ibe_map/decomposed')
    p.add_argument('--image-cache', type=Path, default=ROOT / 'output/e2e_images')
    p.add_argument('--weights', type=Path, default=ROOT / 'output/map/cache/weights/resnet50-0676ba61.pth')
    p.add_argument('--microbatch', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--splits', nargs='+', choices=('train', 'simple', 'hard'),
                   default=['train', 'simple', 'hard'], help='Use train alone for pipeline integration diagnostics')
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
