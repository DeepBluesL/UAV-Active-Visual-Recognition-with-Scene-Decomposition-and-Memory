"""Frozen-protocol repeated training, held-out learning curves, and seed statistics.

All training jobs finish before any hard-test checkpoint is evaluated. Checkpoint
selection is the final episode budget, never the best point on a test curve.
Existing single-seed outputs and observation caches are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from map_baseline.environment import HARD, SIMPLE, TRAIN, Protocol

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ('map', 'memo_map', 'sam_map', 'ibe_map')
POLICIES = ('stationary', 'sweep', 'random') + LEARNED
SEEDS = (42, 43, 44, 45, 46)
METRICS = ('accuracy', 'return_mean', 'distance_m', 'decisions', 'success_rate',
           'stop_rate', 'budget_end_rate', 'generalization_gap_return')
PAIRS = (('memo_map', 'map'), ('sam_map', 'map'), ('ibe_map', 'memo_map'),
         ('ibe_map', 'sam_map'), ('ibe_map', 'map'))
# Two-sided 95% Student-t quantiles, df 1..9; avoids a SciPy dependency.
T975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105,
        5: 2.570581836, 6: 2.446911851, 7: 2.364624252, 8: 2.306004135,
        9: 2.262157163}
SOURCE_FILES = ('map_baseline/environment.py', 'map_baseline/experiment.py',
                'map_baseline/model.py', 'map_baseline/checkpoints.py',
                'ibe_map/model.py', 'ibe_map/experiment.py')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_budgets(episodes, save_every):
    if episodes < 1 or save_every < 1:
        raise ValueError('Episode budget and snapshot interval must be positive')
    return sorted({0, episodes, *range(save_every, episodes + 1, save_every)})


def seed_stats(values, deterministic=False):
    values = [float(v) for v in values]
    if not values or not np.isfinite(values).all():
        raise ValueError('Statistics need finite, nonempty values')
    if deterministic:
        if not np.allclose(values, values[0], rtol=0, atol=1e-12):
            raise ValueError('A deterministic baseline changed across runs')
        return {'mean': values[0], 'std': 0., 'ci95_t': None, 'n': 1,
                'values': [values[0]], 'ddof': 0}
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else None
    ci = None
    if n > 1:
        if n - 1 not in T975:
            raise ValueError('This experiment supports at most 10 repeated seeds')
        margin = T975[n - 1] * std / math.sqrt(n)
        ci = [mean - margin, mean + margin]
    return {'mean': mean, 'std': std, 'ci95_t': ci, 'n': n,
            'values': values, 'ddof': 1 if n > 1 else None}


def cache_identity(path):
    """Validate all cached file bytes and retain the exact training fingerprint."""
    path = Path(path).resolve()
    common = None
    files = {}
    train_hash = hashlib.sha256()
    for scene in sorted(TRAIN + SIMPLE + HARD):
        npz_path = path / 'scenes' / f'{scene:03d}.npz'
        meta_path = npz_path.with_suffix('.json')
        meta_bytes = meta_path.read_bytes()
        meta = json.loads(meta_bytes.decode('utf-8'))
        sig = meta['signature']
        signature = {k: sig[k] for k in ('pretrained_sha256', 'classifier_sha256', 'preprocess')}
        signature['dataset_path'] = sig['source']['dataset_path']
        if common is not None and signature != common:
            raise ValueError(f'Mixed observation identity in {path}')
        common = signature
        actual = sha256(npz_path)
        if actual != meta['npz_sha256']:
            raise ValueError(f'Cache bytes differ from metadata: {npz_path}')
        files[f'{scene:03d}'] = {'npz_sha256': actual, 'metadata_sha256': hashlib.sha256(meta_bytes).hexdigest()}
        if scene in TRAIN:
            train_hash.update(meta_bytes)
    return {'path': str(path), 'observation_signature': common, 'scenes': files,
            'training_fingerprint': train_hash.hexdigest()}


def prepare_manifest(args):
    raw, sam = cache_identity(args.raw_cache), cache_identity(args.sam_cache)
    for key in ('pretrained_sha256', 'classifier_sha256', 'dataset_path'):
        if raw['observation_signature'][key] != sam['observation_signature'][key]:
            raise ValueError(f'Raw/SAM {key} differs')
    if 'scene_decomposition' in raw['observation_signature']['preprocess']:
        raise ValueError('Raw cache contains SAM decomposition')
    if not sam['observation_signature']['preprocess'].get('scene_decomposition'):
        raise ValueError('SAM cache has no decomposition identity')
    if Path(raw['observation_signature']['dataset_path']).resolve() != args.data.resolve():
        raise ValueError('Selected image dataset does not match the observation caches')
    identity = {'schema_version': 1, 'seeds': args.seeds, 'learned_policies': list(LEARNED),
        'episodes': args.episodes, 'save_every': args.save_every,
        'snapshot_budgets': snapshot_budgets(args.episodes, args.save_every),
        'python': str(Path(sys.executable).resolve()), 'device': args.device,
        'training_scenes': TRAIN, 'hard_scenes': HARD, 'simple_scenes': SIMPLE,
        'protocol': Protocol().as_dict(),
        'hyperparameters': {'batch_size': 256, 'environments': 64, 'replay_capacity': 100000,
            'learning_starts': 2048, 'updates_per_vector_step': 1, 'target_update_every': 500,
            'learning_rate': .0001, 'optimizer': 'Adam'},
        'source_sha256': {name: sha256(ROOT / name) for name in SOURCE_FILES},
        'caches': {'raw': raw, 'sam': sam},
        'training_selection': 'final requested episode budget; all training finishes before hard evaluation',
        'randomness': 'learned: training seeds; random baseline: evaluation seeds; sweep/stationary: deterministic',
        'curve_smoothing': 'none; full fixed-grid greedy evaluation at each requested checkpoint'}
    destination = args.out / 'run_manifest.json'
    if destination.exists():
        if read_json(destination) != identity:
            raise ValueError('Run identity changed (source, seeds, budget, cache, or runtime). Use a new output directory.')
    else:
        if args.out.exists() and any(p.name != 'run.log' for p in args.out.iterdir()):
            raise ValueError('Output directory has artifacts but no run_manifest.json; refusing to adopt unknown results')
        write_json(destination, identity)
    return identity


def train_dir(args, seed, policy):
    return args.out / f'seed_{seed}' / policy / 'train'


def model_cache(args, policy):
    return args.sam_cache if policy in ('sam_map', 'ibe_map') else args.raw_cache


def check_checkpoint(path, seed, policy, manifest, requested_checkpoint=None):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    cache = manifest['caches']['sam' if policy in ('sam_map', 'ibe_map') else 'raw']
    expected = {'seed': seed, 'phase': 'full', 'requested_episodes': manifest['episodes'],
        'training_scenes': TRAIN, 'validation_scenes': [], 'protocol': manifest['protocol'],
        'observation_signature': cache['observation_signature'],
        'dataset_fingerprint': cache['training_fingerprint'],
        'checkpoint_selection': 'final requested training budget', **manifest['hyperparameters']}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{path}: mismatched training {key}')
    if config.get('variant', 'map') != policy:
        raise ValueError(f'{path}: incorrect policy identity')
    if Path(config['cache_path']).resolve() != Path(cache['path']).resolve():
        raise ValueError(f'{path}: unexpected cache path')
    if config.get('torch_version') != str(torch.__version__):
        raise ValueError(f'{path}: PyTorch version changed since training')
    count = int(checkpoint['completed_episodes'])
    if requested_checkpoint is not None:
        upper = requested_checkpoint + manifest['hyperparameters']['environments'] - 1
        if count < requested_checkpoint or count > upper or (requested_checkpoint == 0 and count != 0):
            raise ValueError(f'{path}: snapshot episode count {count} does not match {requested_checkpoint}')
    return checkpoint


def validate_training(args, manifest, seed, policy):
    directory = train_dir(args, seed, policy)
    last = directory / 'last.pt'
    summary_path = directory / 'training.json'
    expected_paths = [directory / 'snapshots' / f'episodes_{budget:06d}.pt'
                      for budget in manifest['snapshot_budgets']]
    if not last.exists() or not summary_path.exists() or not all(p.exists() for p in expected_paths):
        return False
    final = check_checkpoint(last, seed, policy, manifest, manifest['episodes'])
    summary = read_json(summary_path)
    if summary['config'] != final['config']:
        raise ValueError(f'Training metadata/checkpoint config mismatch: {directory}')
    for key in ('completed_episodes', 'transitions', 'updates'):
        if summary[key] != final[key]:
            raise ValueError(f'Training metadata/checkpoint {key} mismatch: {directory}')
    previous = (-1, -1, -1)
    for budget, path in zip(manifest['snapshot_budgets'], expected_paths):
        snapshot = check_checkpoint(path, seed, policy, manifest, budget)
        counters = tuple(int(snapshot[key]) for key in ('completed_episodes', 'transitions', 'updates'))
        if any(current < before for current, before in zip(counters, previous)):
            raise ValueError(f'Nonmonotone training counters in snapshot series: {path}')
        if budget == 0 and counters != (0, 0, 0):
            raise ValueError(f'Initial snapshot must precede all training: {path}')
        if counters[1] % manifest['hyperparameters']['environments']:
            raise ValueError(f'Transition count does not match vector environment steps: {path}')
        previous = counters
        if budget == manifest['episodes']:
            import torch
            if counters != tuple(int(final[key]) for key in ('completed_episodes', 'transitions', 'updates')):
                raise ValueError(f'Final snapshot counters differ from final checkpoint: {directory}')
            if set(snapshot['state_dict']) != set(final['state_dict']) or any(
                    not torch.equal(snapshot['state_dict'][k], v) for k, v in final['state_dict'].items()):
                raise ValueError(f'Final snapshot differs from final checkpoint: {directory}')
    return True


def run_command(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('Run: ' + subprocess.list2cmdline([str(s) for s in command]), flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.run([str(s) for s in command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-15:]
        raise RuntimeError(f'Command failed; see {log_path}\n' + '\n'.join(tail))


def train_all(args, manifest):
    for seed in args.seeds:
        for policy in LEARNED:
            if validate_training(args, manifest, seed, policy):
                print(f'Reuse validated training: seed={seed}, policy={policy}', flush=True)
                continue
            directory = train_dir(args, seed, policy)
            # A partial job belongs to this immutable manifest; restarting it
            # overwrites only that job's checkpoints, never an earlier study.
            command = [sys.executable, '-m', 'map_baseline.experiment' if policy == 'map' else 'ibe_map.experiment',
                'train', '--phase', 'full', '--seed', seed, '--episodes', args.episodes,
                '--save-every', args.save_every, '--log-every', args.save_every,
                '--cache', model_cache(args, policy), '--out', directory, '--device', args.device]
            if policy != 'map':
                command.extend(['--variant', policy])
            run_command(command, directory.parent / 'training.log')
            if not validate_training(args, manifest, seed, policy):
                raise ValueError(f'Training did not produce the complete checkpoint series: {directory}')
            print(f'Training complete: seed={seed}, policy={policy}', flush=True)


def require_all_training(args, manifest):
    for seed in args.seeds:
        for policy in LEARNED:
            if not validate_training(args, manifest, seed, policy):
                raise ValueError('Finish every training job before evaluating hard checkpoints: '
                                 f'seed={seed}, policy={policy}')


def curves_all(args, manifest):
    import torch
    from map_baseline.environment import Dataset
    from map_baseline.experiment import aggregate
    from .experiment import SequenceRunner, rollout, validate_cache_for_variant, verify_shared_observations
    from .model import make_model
    require_all_training(args, manifest)
    torch.set_num_threads(4)
    destination = args.out / 'curves.json'
    rows = read_json(destination)['curves'] if destination.exists() else []
    lookup = {}
    for row in rows:
        key = (row['seed'], row['policy'], row['requested_episodes'], row['split'])
        if key in lookup:
            raise ValueError(f'Duplicate curve point: {key}')
        lookup[key] = row
    scenes = sorted(TRAIN + HARD)
    raw, sam = Dataset(args.raw_cache, scenes), Dataset(args.sam_cache, scenes)
    validate_cache_for_variant(raw, 'map')
    validate_cache_for_variant(sam, 'ibe_map')
    verify_shared_observations(raw, sam)
    raw_runner, sam_runner = SequenceRunner(raw, args.device), SequenceRunner(sam, args.device)
    protocol = Protocol(**manifest['protocol'])
    for seed in args.seeds:
        for policy in LEARNED:
            runner = sam_runner if policy in ('sam_map', 'ibe_map') else raw_runner
            model = make_model(policy).to(runner.device)
            for budget in manifest['snapshot_budgets']:
                path = train_dir(args, seed, policy) / 'snapshots' / f'episodes_{budget:06d}.pt'
                digest = sha256(path)
                checkpoint = check_checkpoint(path, seed, policy, manifest, budget)
                if checkpoint['config']['observation_signature'] != runner.data.observation_signature:
                    raise ValueError(f'Curve cache identity mismatch: {path}')
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                for split, group in (('train', TRAIN), ('hard', HARD)):
                    key = (seed, policy, budget, split)
                    if key in lookup:
                        if lookup[key]['checkpoint_sha256'] != digest:
                            raise ValueError(f'Curve checkpoint changed: {key}')
                        continue
                    result = rollout(runner, model, runner.data.indices(group), policy, protocol, seed)
                    row = {'seed': seed, 'policy': policy, 'requested_episodes': budget,
                        'actual_episodes': checkpoint['completed_episodes'], 'split': split,
                        'checkpoint_sha256': digest, **aggregate(result)}
                    rows.append(row)
                    lookup[key] = row
                write_json(destination, {'schema_version': 1, 'curves': rows,
                    'evaluation': 'all 605 starts per context; native greedy policy; unsmoothed'})
                print(f'Curve evaluated: seed={seed}, policy={policy}, episodes={budget}', flush=True)
            del model
    expected = len(args.seeds) * len(LEARNED) * len(manifest['snapshot_budgets']) * 2
    if len(rows) != expected:
        raise ValueError(f'Unexpected curve point count: {len(rows)} versus {expected}')
    del raw_runner, sam_runner, raw, sam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def verification_is_current(directory):
    path = directory / 'verification.json'
    if not path.exists():
        return False
    stamp_path = directory / 'postprocess_identity.json'
    if not stamp_path.exists():
        return False
    stamp = read_json(stamp_path)
    if (stamp.get('verify_results_sha256') != sha256(ROOT / 'ibe_map/verify_results.py')
            or stamp.get('verification_sha256') != sha256(path)):
        return False
    result = read_json(path)
    if result.get('passed') is not True or result['counts']['trajectories'] != 726000:
        raise ValueError(f'Incomplete final verification: {path}')
    identities = {Path(k).resolve(): v for k, v in result['input_sha256'].items()}
    for name in ('results.json', 'evaluation.csv', 'trajectories.jsonl.gz'):
        artifact = (directory / name).resolve()
        if not artifact.exists() or identities.get(artifact) != sha256(artifact):
            return False
    return True


def final_all(args, manifest):
    require_all_training(args, manifest)
    for seed in args.seeds:
        directory = args.out / f'seed_{seed}' / 'evaluation'
        hashes = {p: sha256(train_dir(args, seed, p) / 'last.pt') for p in LEARNED}
        result_path = directory / 'results.json'
        reusable = False
        if result_path.exists():
            result = read_json(result_path)
            if result['protocol']['checkpoint_sha256'] != hashes or result['protocol']['evaluation_seed'] != seed:
                raise ValueError(f'Final evaluation belongs to different models/seed: {result_path}')
            expected_groups = {(s, p) for s in ('train', 'simple', 'hard') for p in POLICIES}
            expected_groups.update(('hard_fixed_3_moves', p) for p in POLICIES if p != 'stationary')
            if {(m['split'], m['policy']) for m in result['metrics']} != expected_groups:
                raise ValueError(f'Final evaluation policy/split mismatch: {result_path}')
            reusable = verification_is_current(directory)
        if reusable:
            print(f'Reuse fully verified final evaluation: seed={seed}', flush=True)
            continue
        if not result_path.exists():
            command = [sys.executable, '-m', 'ibe_map.experiment', 'evaluate', '--out', directory,
                '--raw-cache', args.raw_cache, '--sam-cache', args.sam_cache,
                '--data', args.data, '--device', args.device, '--seed', seed, '--no-report']
            for policy in LEARNED:
                command.extend([f'--{policy.replace("_", "-")}-checkpoint', train_dir(args, seed, policy) / 'last.pt'])
            run_command(command, directory.parent / 'evaluation.log')
        command = [sys.executable, '-m', 'ibe_map.verify_results', '--out', directory,
                   '--raw-cache', args.raw_cache, '--data', args.data]
        verifier_hash = sha256(ROOT / 'ibe_map/verify_results.py')
        run_command(command, directory.parent / 'verification.log')
        if verifier_hash != sha256(ROOT / 'ibe_map/verify_results.py'):
            raise ValueError('Verifier source changed during verification; rerun this stage')
        write_json(directory / 'postprocess_identity.json', {
            'verify_results_sha256': verifier_hash,
            'verification_sha256': sha256(directory / 'verification.json'),
            'note': 'Postprocessing identity only; does not change the locked training identity.'})
        if not verification_is_current(directory):
            raise ValueError(f'Final verification is not current: {directory}')
        print(f'Final evaluation and 726000-trajectory verification complete: seed={seed}', flush=True)


def summarize_runs(final_runs):
    """Aggregate repeated-seed scalars, never pooled episode/scene replicates."""
    if len({r['seed'] for r in final_runs}) != len(final_runs):
        raise ValueError('Duplicate final-run seeds')
    lookups = [{(m['split'], m['policy']): m for m in r['metrics']} for r in final_runs]
    groups = set(lookups[0])
    if any(set(lookup) != groups for lookup in lookups):
        raise ValueError('Repeated runs contain different policy/split groups')
    final_summary = []
    for split, policy in sorted(groups):
        items = [lookup[(split, policy)] for lookup in lookups]
        if len({m['episodes'] for m in items}) != 1:
            raise ValueError(f'Unequal evaluation counts: {split}/{policy}')
        randomness = ('deterministic' if policy in ('stationary', 'sweep') else
                      'evaluation_seed' if policy == 'random' else 'training_seed')
        stats = {metric: seed_stats([m[metric] for m in items], randomness == 'deterministic')
                 for metric in METRICS if all(metric in m for m in items)}
        final_summary.append({'split': split, 'policy': policy, 'randomness': randomness,
            'episodes_per_run': items[0]['episodes'], 'metrics': stats})
    paired = []
    for left, right in PAIRS:
        stats = {}
        for metric in ('accuracy', 'return_mean', 'distance_m', 'generalization_gap_return'):
            if all(metric in lookup[('hard', p)] for lookup in lookups for p in (left, right)):
                stats[metric] = seed_stats([lookup[('hard', left)][metric] - lookup[('hard', right)][metric]
                                           for lookup in lookups])
        paired.append({'left_policy': left, 'right_policy': right, 'split': 'hard', 'metrics': stats})
    return final_summary, paired


def summarize(args, manifest):
    require_all_training(args, manifest)
    curves = read_json(args.out / 'curves.json')['curves']
    expected_keys = {(s, p, b, split) for s in args.seeds for p in LEARNED
                     for b in manifest['snapshot_budgets'] for split in ('train', 'hard')}
    actual_keys = {(r['seed'], r['policy'], r['requested_episodes'], r['split']) for r in curves}
    if actual_keys != expected_keys or len(curves) != len(expected_keys):
        raise ValueError('Curve set is incomplete or duplicated')
    final_runs = []
    for seed in args.seeds:
        directory = args.out / f'seed_{seed}' / 'evaluation'
        if not verification_is_current(directory):
            raise ValueError(f'Final results need independent verification: {directory}')
        result = read_json(directory / 'results.json')
        verification = read_json(directory / 'verification.json')
        expected_hashes = {p: sha256(train_dir(args, seed, p) / 'last.pt') for p in LEARNED}
        if result['protocol']['checkpoint_sha256'] != expected_hashes or result['protocol']['evaluation_seed'] != seed:
            raise ValueError(f'Final evaluation checkpoint/seed mismatch: {directory}')
        for metric in result['metrics']:
            if metric['policy'] not in LEARNED or metric['split'] not in ('train', 'hard'):
                continue
            point = next(r for r in curves if r['seed'] == seed and r['policy'] == metric['policy']
                         and r['split'] == metric['split'] and r['requested_episodes'] == args.episodes)
            for name in ('accuracy', 'return_mean', 'distance_m', 'decisions'):
                if not math.isclose(point[name], metric[name], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f'Final curve/evaluation disagreement: {seed}/{metric["policy"]}/{name}')
        final_runs.append({'seed': seed, 'results_path': str((directory / 'results.json').resolve()),
            'verification_path': str((directory / 'verification.json').resolve()),
            'examples_path': str((directory / 'examples/report.html').resolve()),
            'checkpoint_sha256': expected_hashes, 'metrics': result['metrics'],
            'training': result['training'], 'verification_counts': verification['counts']})
    final_summary, paired = summarize_runs(final_runs)
    total = {key: sum(run['verification_counts'][key] for run in final_runs)
             for key in ('trajectories', 'csv_rows', 'unique_episode_keys', 'metrics_groups')}
    training_records = [run['training'][policy] for run in final_runs for policy in LEARNED]
    training_totals = {'policy_training_runs': len(training_records),
        'requested_episodes_per_run': args.episodes,
        'actual_completed_episodes': sum(r['completed_episodes'] for r in training_records),
        'transitions': sum(r['transitions'] for r in training_records),
        'gradient_updates': sum(r['updates'] for r in training_records),
        'summed_training_seconds': sum(r['elapsed_seconds'] for r in training_records)}
    output = {'schema_version': 1, 'seeds': args.seeds, 'policies': list(POLICIES),
        'learned_policies': list(LEARNED), 'training': training_totals, 'curves': curves, 'final_runs': final_runs,
        'final_summary': final_summary, 'paired_comparisons': paired,
        'verification_total': {'passed': True, **total},
        'protocol': {**manifest['protocol'], 'requested_training_episodes': args.episodes,
            'checkpoint_budgets': manifest['snapshot_budgets'], 'initial_positions': 'all 605 viewpoints per context',
            'curve_evaluation': 'greedy; all 100 training and 25 hard scenes; unsmoothed; after all training completed',
            'interval': 'two-sided Student-t 95% CI across seeds, sample SD (ddof=1); no clipping',
            'paired_interval': 'same-seed method differences; accuracy remains a fraction, not percentage points',
            'deterministic_baselines': 'stationary and sweep counted once; no seed CI',
            'random_baseline': 'different evaluation RNG seed per run; not a trained policy'},
        'limitations': [
            'Five policy training seeds only when the seeds list contains five entries; custom runs may use fewer.',
            'The classifier, ImageNet ResNet50 features, and SAM decomposition are fixed across all policy seeds.',
            'Official pretrained SAM ViT-B is not fine-tuned on UEVAVD; published masks and author fine-tuned weights are unavailable.',
            'Frozen visual backbone, local Q-branch composition, ground-truth boxes, and discrete motion remain reproduction differences.',
            'Seed intervals condition on these fixed contexts and upstream models; they do not estimate uncertainty over new scenes, classifier training, or SAM preprocessing.',
            'The paper smooths its learning curves; these checkpoint curves are shown without smoothing. Exact author seed IDs and epoch cadence were not specified.',
            'Hard-test curves are retrospective diagnostics; no test-based hyperparameter or checkpoint selection is performed.',
            'Paired seed comparisons are descriptive with only a few seeds and no multiple-comparison correction.'
        ], 'sources': [{'title': 'UEVAVD arXiv v1, IV-B', 'url': 'https://arxiv.org/html/2411.04348v1#S4.SS2'}],
        'run_manifest': str((args.out / 'run_manifest.json').resolve()),
        'postprocessing_sha256': {name: sha256(ROOT / name) for name in
            ('ibe_map/multiseed.py', 'ibe_map/multiseed_report.py', 'ibe_map/verify_results.py')
            if (ROOT / name).exists()}}
    write_json(args.out / 'summary.json', output)
    print(f'Summary: {(args.out / "summary.json").resolve()}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/ibe_multiseed')
    parser.add_argument('--raw-cache', type=Path, default=ROOT / 'output/map/cache')
    parser.add_argument('--sam-cache', type=Path, default=ROOT / 'output/ibe_map/cache')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--episodes', type=int, default=300000)
    parser.add_argument('--save-every', type=int, default=30000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('all', 'plan', 'train', 'curves', 'final', 'summarize'), default='all')
    args = parser.parse_args()
    if not 1 <= len(args.seeds) <= 10 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use between 1 and 10 distinct seeds')
    if args.episodes < 1 or args.save_every < 1:
        parser.error('Training and snapshot budgets must be positive')
    args.out = args.out.resolve()
    manifest = prepare_manifest(args)
    if args.stage in ('all', 'train'):
        train_all(args, manifest)
    if args.stage in ('all', 'curves'):
        curves_all(args, manifest)
    if args.stage in ('all', 'final'):
        final_all(args, manifest)
    if args.stage in ('all', 'summarize'):
        summarize(args, manifest)


if __name__ == '__main__':
    main()
