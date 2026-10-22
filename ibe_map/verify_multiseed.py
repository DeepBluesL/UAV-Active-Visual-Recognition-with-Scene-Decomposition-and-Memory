"""Independently audit the formal 300k/30k repeated-seed summary.

Uses only the Python standard library. It does not import the runner,
multiseed orchestration module, NumPy, or PyTorch. It writes only
summary_verification.json and never edits the source study artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickletools
import statistics
import sys
import zipfile
from pathlib import Path


LEARNED = ('map', 'memo_map', 'sam_map', 'ibe_map')
POLICIES = ('stationary', 'sweep', 'random') + LEARNED
METRICS = ('accuracy', 'return_mean', 'distance_m', 'decisions', 'success_rate',
           'stop_rate', 'budget_end_rate', 'generalization_gap_return')
PAIRS = (('memo_map', 'map'), ('sam_map', 'map'), ('ibe_map', 'memo_map'),
         ('ibe_map', 'sam_map'), ('ibe_map', 'map'))
T975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105,
        5: 2.570581836, 6: 2.446911851, 7: 2.364624252, 8: 2.306004135,
        9: 2.262157163}
SPLIT_EPISODES = {'train': 60500, 'simple': 15125, 'hard': 15125,
                  'hard_fixed_3_moves': 15125}


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def require_distinct(values, label):
    require(len(set(values)) == len(values), f'Duplicate {label}')


def insert_unique(mapping, key, value, label):
    require(key not in mapping, f'Duplicate {label}: {key}')
    mapping[key] = value


def read_json(path):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise AuditError(f'Cannot parse JSON {path}: {exc}') from exc


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, tol=1e-12):
    try:
        return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def compare_tree(got, expected, where='value'):
    """Compare JSON values, allowing only harmless floating roundoff."""
    if isinstance(expected, dict):
        require(isinstance(got, dict) and set(got) == set(expected), f'{where}: object keys differ')
        for key in expected:
            compare_tree(got[key], expected[key], f'{where}.{key}')
    elif isinstance(expected, list):
        require(isinstance(got, list) and len(got) == len(expected), f'{where}: list length differs')
        for index, (actual, wanted) in enumerate(zip(got, expected)):
            compare_tree(actual, wanted, f'{where}[{index}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(close(got, expected), f'{where}: {got!r} != {expected!r}')
    else:
        require(got == expected, f'{where}: {got!r} != {expected!r}')


def seed_stats(values, deterministic=False):
    values = [float(value) for value in values]
    require(bool(values) and all(math.isfinite(value) for value in values),
            'Statistics need finite, nonempty values')
    if deterministic:
        require(all(close(value, values[0]) for value in values),
                'A deterministic baseline changed across runs')
        return {'mean': values[0], 'std': 0.0, 'ci95_t': None, 'n': 1,
                'values': [values[0]], 'ddof': 0}
    n = len(values)
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values) if n > 1 else None
    ci = None
    if n > 1:
        require(n - 1 in T975, f'No Student-t critical value for n={n}')
        margin = T975[n - 1] * sample_sd / math.sqrt(n)
        ci = [mean - margin, mean + margin]
    return {'mean': mean, 'std': sample_sd, 'ci95_t': ci, 'n': n,
            'values': values, 'ddof': 1 if n > 1 else None}


def expected_groups():
    groups = {(split, policy) for split in ('train', 'simple', 'hard') for policy in POLICIES}
    groups.update(('hard_fixed_3_moves', policy) for policy in POLICIES if policy != 'stationary')
    return groups


def result_metrics(result, where):
    metrics = result.get('metrics')
    require(isinstance(metrics, list) and metrics, f'{where}: missing metrics')
    lookup = {}
    for metric in metrics:
        key = (metric.get('split'), metric.get('policy'))
        insert_unique(lookup, key, metric, f'{where} metric group')
    require(set(lookup) == expected_groups(), f'{where}: policy/split groups are incomplete or unexpected')
    for (split, policy), metric in lookup.items():
        require(metric.get('episodes') == SPLIT_EPISODES[split],
                f'{where}: wrong episode count for {split}/{policy}: {metric.get("episodes")}')
    for policy in POLICIES:
        expected_gap = lookup[('train', policy)]['return_mean'] - lookup[('hard', policy)]['return_mean']
        require(close(lookup[('hard', policy)].get('generalization_gap_return'), expected_gap),
                f'{where}: generalization gap mismatch for {policy}')
    return lookup


def indexed_hashes(verification):
    raw = verification.get('input_sha256')
    require(isinstance(raw, dict), 'verification.json has no input_sha256 map')
    normalized = {}
    for name, value in raw.items():
        key = str(Path(name).resolve())
        require(key not in normalized, f'verification.json repeats input path {key}')
        normalized[key] = value
    return normalized


def checkpoint_counters(path):
    """Read scalar training counters from torch.save's pickle without loading tensors.

    The pickle is interpreted as opcodes only; no pickle globals are imported or
    executed. This supports the ZIP serialization emitted by torch.save.
    """
    integer_ops = {'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4'}
    wanted = {'completed_episodes', 'transitions', 'updates'}
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith('/data.pkl') or name == 'data.pkl']
        require(len(members) == 1, f'{path}: expected one torch data.pkl member')
        ops = list(pickletools.genops(archive.read(members[0])))
    counters = {}
    memo_ops = {'BINPUT', 'LONG_BINPUT', 'PUT'}
    for index, (opcode, argument, _) in enumerate(ops[:-1]):
        if opcode.name not in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'SHORT_BINSTRING', 'BINSTRING'):
            continue
        if argument not in wanted:
            continue
        value_index = index + 1
        while value_index < len(ops) and ops[value_index][0].name in memo_ops:
            value_index += 1
        require(value_index < len(ops), f'{path}: truncated checkpoint field {argument}')
        value_op, value, _ = ops[value_index]
        require(value_op.name in integer_ops and isinstance(value, int),
                f'{path}: checkpoint field {argument} is not serialized as a scalar integer')
        counters[argument] = value
    require(set(counters) == wanted, f'{path}: missing checkpoint counters {wanted - set(counters)}')
    return counters


def verify_run(out, seed, final_run, source_hashes, manifest):
    directory = out / f'seed_{seed}' / 'evaluation'
    result_path, verification_path = directory / 'results.json', directory / 'verification.json'
    csv_path, trace_path = directory / 'evaluation.csv', directory / 'trajectories.jsonl.gz'
    result = read_json(result_path)
    verification = read_json(verification_path)
    require(verification.get('passed') is True, f'seed {seed}: verification did not pass')
    expected_counts = {'metrics_groups': 27, 'trajectories': 726000, 'csv_rows': 726000,
                       'unique_episode_keys': 726000}
    for name, value in expected_counts.items():
        require(verification.get('counts', {}).get(name) == value,
                f'seed {seed}: verification count {name} is not {value}')
    require(verification.get('checks') and all(value is True for value in verification['checks'].values()),
            f'seed {seed}: one or more trajectory-verification checks are false')

    hashes = indexed_hashes(verification)
    for artifact in (result_path, csv_path, trace_path):
        require(artifact.is_file(), f'seed {seed}: missing verified artifact {artifact}')
        actual = sha256(artifact)
        key = str(artifact.resolve())
        require(hashes.get(key) == actual, f'seed {seed}: SHA256 mismatch for {artifact.name}')
        source_hashes[key] = actual
    source_hashes[str(verification_path.resolve())] = sha256(verification_path)

    require(result.get('protocol', {}).get('evaluation_seed') == seed,
            f'seed {seed}: result has a different evaluation seed')
    expected_checkpoints = {}
    require(isinstance(result.get('training'), dict), f'seed {seed}: results lack training metadata')
    for policy in LEARNED:
        train_directory = out / f'seed_{seed}' / policy / 'train'
        checkpoint = train_directory / 'last.pt'
        require(checkpoint.is_file(), f'seed {seed}: missing final checkpoint for {policy}')
        expected_checkpoints[policy] = sha256(checkpoint)
        counters = checkpoint_counters(checkpoint)
        training = read_json(train_directory / 'training.json')
        config = training.get('config', {})
        require(config.get('seed') == seed and config.get('phase') == 'full'
                and config.get('requested_episodes') == 300000
                and config.get('validation_scenes') == [],
                f'seed {seed}/{policy}: training manifest does not identify a full 300k run')
        expected_train_scenes = [base + offset for base in (0, 30, 60, 90, 120)
                                 for offset in range(1, 21)]
        require(config.get('training_scenes') == expected_train_scenes,
                f'seed {seed}/{policy}: training scene split differs from the fixed 100-scene split')
        require(config.get('variant', 'map') == policy,
                f'seed {seed}/{policy}: training manifest policy mismatch')
        cache_name = 'sam' if policy in ('sam_map', 'ibe_map') else 'raw'
        cache = manifest.get('caches', {}).get(cache_name, {})
        expected_config = {'protocol': manifest.get('protocol'),
            'checkpoint_selection': 'final requested training budget',
            'observation_signature': cache.get('observation_signature'),
            'dataset_fingerprint': cache.get('training_fingerprint'),
            'cache_path': cache.get('path')}
        expected_config.update(manifest.get('hyperparameters', {}))
        for name, value in expected_config.items():
            require(config.get(name) == value,
                    f'seed {seed}/{policy}: checkpoint training identity differs at {name}')
        for name, value in counters.items():
            require(training.get(name) == value,
                    f'seed {seed}/{policy}: training.json/checkpoint {name} mismatch')
        compare_tree(result['training'].get(policy), training,
                     f'seed {seed}/{policy} results training metadata')
    require(result['protocol'].get('checkpoint_sha256') == expected_checkpoints,
            f'seed {seed}: final result checkpoint hashes do not match last.pt files')
    require(final_run.get('checkpoint_sha256') == expected_checkpoints,
            f'seed {seed}: summary checkpoint hashes differ from final checkpoints')

    lookup = result_metrics(result, f'seed {seed} results')
    require(final_run.get('seed') == seed, f'summary has the wrong seed in final_runs for {seed}')
    require(Path(final_run.get('results_path', '')).resolve() == result_path.resolve(),
            f'seed {seed}: summary points at another results.json')
    require(Path(final_run.get('verification_path', '')).resolve() == verification_path.resolve(),
            f'seed {seed}: summary points at another verification.json')
    compare_tree(final_run.get('metrics'), result['metrics'], f'summary final_runs seed {seed} metrics')
    compare_tree(final_run.get('verification_counts'), verification['counts'],
                 f'summary final_runs seed {seed} verification_counts')
    return result, verification, lookup


def expected_final_summaries(run_lookups, seeds):
    groups = set(run_lookups[0])
    require(all(set(lookup) == groups for lookup in run_lookups),
            'Final runs contain different metric groups')
    result = []
    for split, policy in sorted(groups):
        items = [lookup[(split, policy)] for lookup in run_lookups]
        require(len({item['episodes'] for item in items}) == 1,
                f'Unequal evaluation counts for {split}/{policy}')
        randomness = ('deterministic' if policy in ('stationary', 'sweep') else
                      'evaluation_seed' if policy == 'random' else 'training_seed')
        metric_stats = {}
        for metric in METRICS:
            if all(metric in item for item in items):
                metric_stats[metric] = seed_stats([item[metric] for item in items],
                                                  deterministic=randomness == 'deterministic')
        result.append({'split': split, 'policy': policy, 'randomness': randomness,
                       'episodes_per_run': items[0]['episodes'], 'metrics': metric_stats})

    paired = []
    for left, right in PAIRS:
        stats = {}
        for metric in ('accuracy', 'return_mean', 'distance_m', 'generalization_gap_return'):
            if all(metric in lookup[('hard', policy)]
                   for lookup in run_lookups for policy in (left, right)):
                differences = [lookup[('hard', left)][metric] - lookup[('hard', right)][metric]
                               for lookup in run_lookups]
                stats[metric] = seed_stats(differences)
        paired.append({'left_policy': left, 'right_policy': right, 'split': 'hard', 'metrics': stats})
    return result, paired


def verify_curves(out, summary, seeds, final_lookups, source_hashes):
    curves_path = out / 'curves.json'
    curves_doc = read_json(curves_path)
    rows = curves_doc.get('curves')
    require(isinstance(rows, list), 'curves.json has no curves list')
    budgets = summary.get('protocol', {}).get('checkpoint_budgets')
    require(budgets == list(range(0, 300001, 30000)),
            'Expected fixed 0..300000 episode checkpoint schedule at 30000 intervals')
    require(summary.get('protocol', {}).get('requested_training_episodes') == 300000,
            'Expected the formal 300000-episode training budget')
    expected = {(seed, policy, budget, split) for seed in seeds for policy in LEARNED
                for budget in budgets for split in ('train', 'hard')}
    lookup = {}
    for row in rows:
        key = (row.get('seed'), row.get('policy'), row.get('requested_episodes'), row.get('split'))
        insert_unique(lookup, key, row, 'curve point')
    require(set(lookup) == expected and len(rows) == len(expected),
            f'Curve set incomplete: got {len(rows)}, expected {len(expected)}')

    for seed in seeds:
        for policy in LEARNED:
            previous_counters = {'completed_episodes': 0, 'transitions': 0, 'updates': 0}
            for budget in budgets:
                train = lookup[(seed, policy, budget, 'train')]
                hard = lookup[(seed, policy, budget, 'hard')]
                require(train.get('episodes') == SPLIT_EPISODES['train'],
                        f'Wrong train curve count for {seed}/{policy}/{budget}')
                require(hard.get('episodes') == SPLIT_EPISODES['hard'],
                        f'Wrong hard curve count for {seed}/{policy}/{budget}')
                for row in (train, hard):
                    actual = row.get('actual_episodes')
                    require(isinstance(actual, int) and budget <= actual <= budget + 63,
                            f'Invalid actual checkpoint episode count for {seed}/{policy}/{budget}: {actual}')
                    require(isinstance(row.get('checkpoint_sha256'), str)
                            and len(row['checkpoint_sha256']) == 64
                            and all(char in '0123456789abcdef' for char in row['checkpoint_sha256'].lower()),
                            f'Invalid checkpoint SHA256 in curve row {seed}/{policy}/{budget}')
                require(train['actual_episodes'] == hard['actual_episodes'],
                        f'Train/hard curve checkpoints have different actual counts: {seed}/{policy}/{budget}')
                require(train['checkpoint_sha256'] == hard['checkpoint_sha256'],
                        f'Train/hard curve rows reference different checkpoints: {seed}/{policy}/{budget}')
                checkpoint = out / f'seed_{seed}' / policy / 'train' / 'snapshots' / f'episodes_{budget:06d}.pt'
                require(checkpoint.is_file(), f'Missing curve checkpoint: {checkpoint}')
                actual_hash = sha256(checkpoint)
                require(train['checkpoint_sha256'] == actual_hash,
                        f'Curve checkpoint hash mismatch: {seed}/{policy}/{budget}')
                counters = checkpoint_counters(checkpoint)
                require(counters['completed_episodes'] == train['actual_episodes'],
                        f'Curve actual episode count differs from checkpoint: {seed}/{policy}/{budget}')
                require(all(counters[name] >= previous_counters[name] for name in previous_counters),
                        f'Checkpoint counters went backwards: {seed}/{policy}/{budget}')
                previous_counters = counters
                source_hashes[str(checkpoint.resolve())] = actual_hash

    for seed_index, seed in enumerate(seeds):
        metrics = final_lookups[seed_index]
        for policy in LEARNED:
            curve = lookup[(seed, policy, 300000, 'train')]
            result = metrics[('train', policy)]
            for field in ('accuracy', 'return_mean', 'distance_m'):
                require(close(curve[field], result[field]),
                        f'Final train curve/result mismatch: {seed}/{policy}/{field}')
            curve = lookup[(seed, policy, 300000, 'hard')]
            result = metrics[('hard', policy)]
            for field in ('accuracy', 'return_mean', 'distance_m'):
                require(close(curve[field], result[field]),
                        f'Final hard curve/result mismatch: {seed}/{policy}/{field}')
    source_hashes[str(curves_path.resolve())] = sha256(curves_path)
    return rows


def audit(out):
    out = Path(out).resolve()
    summary_path = out / 'summary.json'
    manifest_path = out / 'run_manifest.json'
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    source_hashes = {str(summary_path.resolve()): sha256(summary_path),
                     str(manifest_path.resolve()): sha256(manifest_path)}
    seeds = summary.get('seeds')
    require(isinstance(seeds, list) and 2 <= len(seeds) <= 10,
            'Expected 2..10 training seeds')
    require_distinct(seeds, 'training seed')
    require(manifest.get('seeds') == seeds and manifest.get('episodes') == 300000
            and manifest.get('save_every') == 30000,
            'run_manifest.json does not match the five-seed 300k/30k study protocol')
    require(summary.get('learned_policies') == list(LEARNED)
            and summary.get('policies') == list(POLICIES), 'Unexpected policy set in summary')
    require(summary.get('verification_total', {}).get('passed') is True,
            'summary verification_total is not marked passed')
    final_runs = summary.get('final_runs')
    require(isinstance(final_runs, list) and len(final_runs) == len(seeds),
            'summary final_runs does not cover every seed exactly once')
    by_seed = {}
    for run in final_runs:
        seed = run.get('seed')
        insert_unique(by_seed, seed, run, 'final run seed')
    require(set(by_seed) == set(seeds), 'summary final_runs seeds differ from summary seeds')

    results, verifications, lookups = [], [], []
    for seed in seeds:
        result, verification, lookup = verify_run(out, seed, by_seed[seed], source_hashes, manifest)
        results.append(result)
        verifications.append(verification)
        lookups.append(lookup)
    compare_tree(summary.get('final_runs'), [by_seed[seed] for seed in seeds], 'final_runs ordering')
    for seed, run, result in zip(seeds, [by_seed[s] for s in seeds], results):
        compare_tree(run.get('metrics'), result['metrics'], f'seed {seed} raw metrics')

    final_summary, paired = expected_final_summaries(lookups, seeds)
    compare_tree(summary.get('final_summary'), final_summary, 'independent final_summary')
    compare_tree(summary.get('paired_comparisons'), paired, 'independent paired_comparisons')

    training_records = [result['training'][policy] for result in results for policy in LEARNED]
    expected_training = {'policy_training_runs': len(training_records),
        'requested_episodes_per_run': 300000,
        'actual_completed_episodes': sum(r['completed_episodes'] for r in training_records),
        'transitions': sum(r['transitions'] for r in training_records),
        'gradient_updates': sum(r['updates'] for r in training_records),
        'summed_training_seconds': sum(r['elapsed_seconds'] for r in training_records)}
    compare_tree(summary.get('training'), expected_training, 'independent training totals')

    curves = verify_curves(out, summary, seeds, lookups, source_hashes)
    totals = {'trajectories': 726000 * len(seeds), 'csv_rows': 726000 * len(seeds),
              'unique_episode_keys': 726000 * len(seeds), 'metrics_groups': 27 * len(seeds)}
    calculated = {key: sum(int(v['counts'][key]) for v in verifications) for key in totals}
    require(calculated == totals, f'Independent verification totals mismatch: {calculated}')
    compare_tree(summary.get('verification_total'), {'passed': True, **totals}, 'verification_total')
    source_hashes[str((out / 'summary.json').resolve())] = sha256(summary_path)
    source_hashes[str((out / 'run_manifest.json').resolve())] = sha256(manifest_path)
    return {'schema_version': 1, 'passed': True, 'seeds': seeds,
            'counts': {'runs': len(seeds), 'curve_points': len(curves),
                       'verified_seed_runs': len(results), **calculated},
            'checks': {'run_manifest_seed_budget': True, 'per_seed_verification_passed': True,
                       'verified_result_csv_trace_hashes': True, 'final_raw_metrics_match': True,
                       'final_summary_and_paired_statistics_recomputed': True,
                       'all_curve_points_and_checkpoint_hashes': True,
                       'final_curve_metrics_match_results': True,
                       'verification_totals_match': True},
            'input_sha256': source_hashes,
            'note': 'Trajectory contents were not reparsed; each seed verification.json was checked and its results/CSV/trace SHA256 values were recomputed.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('output/ibe_multiseed'))
    args = parser.parse_args(argv)
    destination = (args.out / 'summary_verification.json').resolve()
    try:
        report = audit(args.out)
    except Exception as exc:
        report = {'schema_version': 1, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        write_json(destination, report)
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    write_json(destination, report)
    print(json.dumps({'passed': True, **report['counts'], 'verification': str(destination)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
