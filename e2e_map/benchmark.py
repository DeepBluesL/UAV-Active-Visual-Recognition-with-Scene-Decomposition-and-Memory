"""Isolated GPU comparisons on one immutable real MAP checkpoint and replay.

Run only after the formal GPU job and its evaluation have finished. Each
candidate executes in a separate process, so an OOM cannot contaminate another
candidate or the formal training result. Results measure acting plus a full TD
optimizer update on fixed recorded observations, not policy quality.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from map_baseline.environment import Dataset
from map_baseline.features import sha256
from ibe_map.experiment import PrefixReplay
from .experiment import td_loss
from .images import RGBImageStore
from .performance import PerformanceModel, PerformanceOptions, PerformanceRunner

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    'baseline_full64': PerformanceOptions('full', 64),
    'none64': PerformanceOptions('none', 64),
    'none128': PerformanceOptions('none', 128),
    'none256': PerformanceOptions('none', 256),
    'late128': PerformanceOptions('late', 128),
    'full64_gpu_pixels': PerformanceOptions('full', 64, True),
    'none128_gpu_pixels': PerformanceOptions('none', 128, True),
}
DEFAULT_CANDIDATES = ['baseline_full64', 'none64', 'none128', 'none256', 'none128_gpu_pixels']


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def make_fixture(saved, count, batch_size=256):
    """Sample identically from the checkpoint's real, uniformly sampled replay."""
    if batch_size != 256 or saved['config']['batch_size'] != batch_size:
        raise ValueError('The benchmark fixes the effective replay batch at 256')
    if saved['replay_size'] < batch_size or count < 1:
        raise ValueError('Insufficient replay or invalid benchmark count')
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(saved['rng'])
    return np.stack([rng.integers(saved['replay_size'], size=batch_size) for _ in range(count)])


def gradient_vector(model):
    """Include every parameter in a stable order, with zero for absent grads."""
    return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten().cpu()
                      for p in model.parameters()])


def compare_correctness(reference, candidate):
    a, b = reference['gradient'].double(), candidate['gradient'].double()
    actions_a, actions_b = np.asarray(reference['actions']), np.asarray(candidate['actions'])
    if (a.shape != b.shape or not a.numel() or actions_a.shape != actions_b.shape
            or actions_a.ndim != 1 or not len(actions_a)):
        return {'passed': False, 'reason': 'Gradient or acting-output shapes do not match'}
    if (not math.isfinite(reference['loss']) or not math.isfinite(candidate['loss'])
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()
            or not np.isfinite(actions_a).all() or not np.isfinite(actions_b).all()):
        return {'passed': False, 'reason': 'Non-finite loss, gradient or acting output'}
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    difference = torch.linalg.vector_norm(a - b)
    cosine = float(torch.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else float(norm_a == norm_b)
    loss_diff = abs(candidate['loss'] - reference['loss'])
    relative = float(difference / norm_a) if norm_a else float(difference)
    result = {'loss_absolute_difference': loss_diff,
              'loss_relative_difference': loss_diff / max(abs(reference['loss']), 1e-12),
              'gradient_relative_l2': relative, 'gradient_cosine': cosine,
              'gradient_max_absolute_difference': float((a - b).abs().max()),
              'acting_action_agreement': float(np.mean(actions_a == actions_b)),
              'thresholds': {'loss_rtol': .005, 'loss_atol': 1e-6,
                             'gradient_relative_l2_max': .02, 'gradient_cosine_min': .999,
                             'acting_action_agreement_min': 1.0}}
    result['passed'] = (loss_diff <= 1e-6 + .005 * abs(reference['loss']) and relative <= .02
                        and cosine >= .999 and result['acting_action_agreement'] == 1.)
    return result


def validate_input(saved):
    config = saved['config']
    if config['variant'] != 'map' or config['batch_size'] != 256 or config['environments'] != 64:
        raise ValueError('Expected the current MAP batch256/env64 training checkpoint')
    if config.get('phase') != 'full' or config.get('requested_episodes') != 300000:
        raise ValueError('Benchmark input must be the full 300000-episode MAP experiment')
    if saved['completed'] < config['requested_episodes']:
        raise ValueError('Benchmark input is incomplete; wait for the current MAP run to finish')
    if 'bfloat16' not in config['precision']:
        raise ValueError('Expected the BF16 training checkpoint')


def validate_limits(warmup, steps):
    if not 1 <= warmup <= 50 or not 2 <= steps <= 200:
        raise ValueError('Use 1..50 warmup and 2..200 measured updates for this short benchmark')


def load_replay(saved):
    capacity = len(saved['replay']['histories'])
    replay = PrefixReplay(capacity, saved['config']['protocol']['max_decisions'])
    for name, value in saved['replay'].items():
        setattr(replay, name, value.numpy().copy())
    replay.size, replay.pos = saved['replay_size'], saved['replay_position']
    return replay


def candidate_worker(args):
    """One process owns all CUDA allocations for one candidate only."""
    run_name = args.run_name or args.candidate
    out = args.out / run_name
    if out.exists() and (out / 'result.json').exists():
        raise ValueError('Candidate output already exists; refuse stale-result reuse')
    out.mkdir(parents=True, exist_ok=True)
    options = CANDIDATES[args.candidate]
    result = {'candidate': args.candidate, 'run_name': run_name, 'options': asdict(options), 'status': 'started'}
    started = time.perf_counter()
    try:
        validate_limits(args.warmup, args.steps)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required for this benchmark')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        manifest = json.loads((args.out / 'input_manifest.json').read_text(encoding='utf8'))
        if args.candidate not in manifest['candidates'] or args.warmup != manifest['warmup'] or args.steps != manifest['steps']:
            raise ValueError('Worker options differ from the immutable benchmark manifest')
        for name, expected in manifest['source_sha256'].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f'Benchmark source changed between candidates: {name}')
        checkpoint = args.out / 'input_resume.pt'
        if sha256(checkpoint) != manifest['resume_sha256']:
            raise ValueError('Immutable benchmark input changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = saved['config']
        validate_input(saved)
        samples = make_fixture(saved, args.warmup + args.steps + 1)
        sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
        if sample_hash != manifest['sample_sha256']:
            raise ValueError('Replay sampling differs from the immutable benchmark fixture')
        data = Dataset(Path(config['cache_path']), config['training_scenes'])
        if data.fingerprint != config['dataset_fingerprint']:
            raise ValueError('Recorded observation cache no longer matches the real checkpoint')
        images = RGBImageStore(data, args.data, args.decomposed, args.image_cache, 'map',
                               observation_cache=Path(config['cache_path']), verbose=False)
        if images.fingerprint != config['rgb_cache_fingerprint']:
            raise ValueError('Recorded RGB pixels no longer match the real checkpoint')
        runner = PerformanceRunner(data, images, 'cuda', gpu_pixels=options.gpu_pixels,
                                   pixel_load_batch=options.pixel_load_batch)
        model = PerformanceModel('map', args.weights, microbatch=options.microbatch,
                                  checkpoint_mode=options.checkpoint_mode).to('cuda', memory_format=torch.channels_last)
        target = copy.deepcopy(model).eval().requires_grad_(False)
        runner.enable_target_cache(target)
        replay = load_replay(saved)
        actor_history = saved['environment']['histories'].numpy().copy()
        actor_lengths = saved['environment']['decision'].numpy().copy() + 1
        gamma = config['protocol']['gamma']

        def restore():
            model.load_state_dict(saved['model'])
            target.load_state_dict(saved['target'])
            runner.load_target_cache_state(saved.get('target_feature_cache'))
            model.zero_grad(set_to_none=True)

        def act():
            model.eval()
            actions = runner.choose(model, actor_history, actor_lengths)
            model.train()
            return actions

        # Correctness is measured from exactly the same weights, target cache,
        # real replay transitions and actor prefixes, before any timed updates.
        restore()
        actions = act()
        loss = td_loss(runner, model, target, replay, samples[0], gamma)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite correctness loss')
        loss.backward()
        gradients = gradient_vector(model)
        if not torch.isfinite(gradients).all():
            raise RuntimeError('Non-finite correctness gradient')
        correctness = {'loss': float(loss.detach()), 'gradient': gradients, 'actions': actions.tolist()}
        torch.save(correctness, out / 'correctness.pt')
        result['correctness_sha256'] = sha256(out / 'correctness.pt')
        del loss, gradients, correctness
        restore()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        scaler.load_state_dict(saved['scaler'])
        skipped, updates = 0, saved['updates']

        def update(sample, force_cold=False):
            nonlocal skipped, updates
            if force_cold:
                runner.invalidate_target()
            act()
            optimizer.zero_grad(set_to_none=True)
            value = td_loss(runner, model, target, replay, sample, gamma)
            if not torch.isfinite(value):
                raise RuntimeError('Non-finite timed TD loss')
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
            else:
                updates += 1
                if updates % config['target_update_every'] == 0:
                    target.load_state_dict(model.state_dict())
                    runner.invalidate_target()
            return float(value.detach())

        for index in range(args.warmup):
            update(samples[index + 1])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_allocated = torch.cuda.memory_allocated()
        initial_reserved = torch.cuda.memory_reserved()
        begin = time.perf_counter()
        losses = [update(samples[index + args.warmup + 1]) for index in range(args.steps)]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        result.update({'status': 'ok', 'checkpoint_sha256': manifest['resume_sha256'],
            'sample_sha256': sample_hash, 'effective_batch_size': 256, 'environments': 64,
            'warmup_updates': args.warmup, 'measured_updates': args.steps, 'measured_seconds': seconds,
            'seconds_per_update': seconds / args.steps, 'updates_per_second': args.steps / seconds,
            'replay_transitions_per_second': args.steps * 256 / seconds,
            'initial_allocated_bytes': initial_allocated, 'initial_reserved_bytes': initial_reserved,
            'warm_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'warm_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'pixel_cache_bytes': runner.pixel_cache.bytes if runner.pixel_cache is not None else 0,
            'amp_skipped_updates': skipped, 'losses': losses, 'gpu': torch.cuda.get_device_name(),
            'torch_version': str(torch.__version__), 'source_sha256': manifest['source_sha256'],
            'scope': '64-prefix acting + full current/next/target TD backward + Adam; fixed real replay/actor observations; excludes env transitions and disk writes',
            'target_cache': 'same saved cache initially, retained between updates, invalidated on original target sync schedule'})
        # Separately report a cold target-cache update. It is not mixed into
        # the warm measurement or used to choose a best-looking timing sample.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cold_begin = time.perf_counter()
        cold_loss = update(samples[-1], force_cold=True)
        torch.cuda.synchronize()
        result['cold_target_update_seconds'] = time.perf_counter() - cold_begin
        result['cold_target_loss'] = cold_loss
        result['cold_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['cold_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['peak_allocated_bytes'] = max(result['warm_peak_allocated_bytes'], result['cold_peak_allocated_bytes'])
        result['peak_reserved_bytes'] = max(result['warm_peak_reserved_bytes'], result['cold_peak_reserved_bytes'])
        result['amp_skipped_updates'] = skipped
        result['all_parameters_finite'] = all(bool(torch.isfinite(v).all()) for v in model.parameters())
        result['batchnorm_running_stats_unchanged'] = all(
            torch.equal(value.detach().cpu(), saved['model'][name])
            for name, value in model.state_dict().items()
            if 'running_' in name or 'num_batches_tracked' in name)
    except torch.OutOfMemoryError as error:
        result.update({'status': 'oom', 'error': str(error),
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    except Exception as error:
        result.update({'status': 'error', 'error': repr(error), 'traceback': traceback.format_exc()})
    result['worker_elapsed_seconds'] = time.perf_counter() - started
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)
    return result


def run_sweep(args):
    validate_limits(args.warmup, args.steps)
    if 'baseline_full64' not in args.candidates:
        raise ValueError('Include baseline_full64 for the numerical and speed comparison')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Benchmark output must be a new or empty directory; existing files are never reused')
    args.out.mkdir(parents=True, exist_ok=True)
    # Copy one atomic training checkpoint once. Every candidate then reads the
    # exact same immutable local file; the source is never modified.
    shutil.copyfile(args.resume_checkpoint, args.out / 'input_resume.pt')
    saved = torch.load(args.out / 'input_resume.pt', map_location='cpu', weights_only=True)
    validate_input(saved)
    samples = make_fixture(saved, args.warmup + args.steps + 1)
    sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
    del saved, samples
    manifest = {'source_checkpoint': str(args.resume_checkpoint.resolve()),
                'resume_sha256': sha256(args.out / 'input_resume.pt'), 'warmup': args.warmup,
                'steps': args.steps, 'candidates': args.candidates, 'sample_sha256': sample_hash,
                'source_sha256': {name: sha256(ROOT / name) for name in
                    ('e2e_map/benchmark.py', 'e2e_map/performance.py', 'e2e_map/model.py', 'e2e_map/experiment.py',
                     'e2e_map/images.py', 'map_baseline/environment.py', 'map_baseline/features.py',
                     'map_baseline/model.py', 'ibe_map/model.py', 'ibe_map/experiment.py')}}
    write_json(args.out / 'input_manifest.json', manifest)
    results = []
    ordered = [(name, name) for name in ['baseline_full64'] + [n for n in args.candidates if n != 'baseline_full64']]
    ordered.append(('baseline_full64', 'baseline_full64_repeat'))
    for name, run_name in ordered:
        folder = args.out / run_name
        folder.mkdir(exist_ok=True)
        command = [sys.executable, '-m', 'e2e_map.benchmark', '--worker', '--candidate', name,
                   '--out', str(args.out), '--warmup', str(args.warmup), '--steps', str(args.steps),
                   '--run-name', run_name,
                   '--data', str(args.data), '--decomposed', str(args.decomposed),
                   '--image-cache', str(args.image_cache), '--weights', str(args.weights)]
        with (folder / 'worker.log').open('w', encoding='utf8') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text(encoding='utf8'))
            if result.get('candidate') != name or result.get('run_name') != run_name or result.get('options') != asdict(CANDIDATES[name]):
                raise ValueError(f'Worker result identity mismatch: {run_name}')
            if result['status'] == 'ok' and any(result.get(key) != manifest[key] for key in ('sample_sha256', 'source_sha256')):
                raise ValueError(f'Worker source or replay identity mismatch: {run_name}')
            if result['status'] == 'ok' and result.get('checkpoint_sha256') != manifest['resume_sha256']:
                raise ValueError(f'Worker checkpoint identity mismatch: {run_name}')
            if result['status'] == 'ok' and sha256(folder / 'correctness.pt') != result['correctness_sha256']:
                raise ValueError(f'Worker correctness artifact changed: {run_name}')
        else:
            result = {'candidate': name, 'run_name': run_name, 'status': 'process_error', 'exit_code': process.returncode}
        results.append(result)
        print(json.dumps({key: value for key, value in result.items()
                          if key in ('candidate', 'status', 'seconds_per_update', 'peak_allocated_bytes', 'error')}), flush=True)
    reference = next(row for row in results if row['candidate'] == 'baseline_full64')
    baselines = [row for row in results if row['candidate'] == 'baseline_full64' and row['status'] == 'ok']
    baseline_seconds = float(np.mean([row['seconds_per_update'] for row in baselines])) if baselines else None
    baseline_drift = ((max(row['seconds_per_update'] for row in baselines) - min(row['seconds_per_update'] for row in baselines))
                      / baseline_seconds) if len(baselines) == 2 else None
    if reference['status'] == 'ok':
        gradients = torch.load(args.out / 'baseline_full64' / 'correctness.pt', weights_only=True)
        for row in results:
            if row['status'] != 'ok':
                continue
            comparison = torch.load(args.out / row['run_name'] / 'correctness.pt', weights_only=True)
            row['correctness'] = compare_correctness(gradients, comparison)
            row['speedup_vs_baseline'] = baseline_seconds / row['seconds_per_update']
            del comparison
        del gradients
    eligible = [row for row in results if row['status'] == 'ok' and row.get('correctness', {}).get('passed')
                and row['amp_skipped_updates'] == 0 and row['all_parameters_finite']
                and row['batchnorm_running_stats_unchanged']]
    timing_stable = baseline_drift is not None and baseline_drift <= .10
    best = min(eligible, key=lambda row: baseline_seconds if row['candidate'] == 'baseline_full64' else row['seconds_per_update']) if eligible and timing_stable else None
    summary = {'manifest': manifest, 'results': results,
               'fastest_verified_candidate': best['candidate'] if best else None,
               'baseline_repeat': {'runs': len(baselines), 'mean_seconds_per_update': baseline_seconds,
                                   'relative_difference': baseline_drift, 'stable_within_10_percent': timing_stable},
               'interpretation': 'Throughput and memory only; this short fixed-replay benchmark is not an accuracy or return experiment.'}
    write_json(args.out / 'summary.json', summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidates', nargs='+', choices=list(CANDIDATES), default=DEFAULT_CANDIDATES)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--data', type=Path, default=ROOT / 'dataset')
    p.add_argument('--decomposed', type=Path, default=ROOT / 'output/ibe_map/decomposed')
    p.add_argument('--image-cache', type=Path, default=ROOT / 'output/e2e_images')
    p.add_argument('--weights', type=Path, default=ROOT / 'output/map/cache/weights/resnet50-0676ba61.pth')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--candidate', choices=list(CANDIDATES), help=argparse.SUPPRESS)
    p.add_argument('--run-name', help=argparse.SUPPRESS)
    return p


if __name__ == '__main__':
    p = parser(); args = p.parse_args()
    if args.worker:
        if not args.candidate:
            p.error('--worker requires --candidate')
        candidate_worker(args)
    else:
        if args.resume_checkpoint is None:
            p.error('--resume-checkpoint is required')
        run_sweep(args)
