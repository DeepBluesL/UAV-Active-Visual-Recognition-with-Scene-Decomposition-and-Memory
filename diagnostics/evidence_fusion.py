"""Compare recognition readouts on fixed, already acquired trajectories.

Run: python -m diagnostics.evidence_fusion
Labels are used only by scoring, never by predict_readouts. No policy, stopping
rule, classifier or reward is changed. The oracle selects a correct top-1 view
using the label; it is not an upper bound on arbitrary feature/probability fusion.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('Hatchback', 'Pickup', 'Sedan', 'Sportscar', 'SUV')
BLOCK_CLASSES = ('Hatchback', 'Pickup', 'Sportscar', 'Sedan', 'SUV')
METHODS = ('final', 'max_confidence', 'mean_unique', 'mean_visits')
POLICIES = ('stationary', 'sweep', 'random', 'map', 'memo_map', 'sam_map', 'ibe_map')
DETERMINISTIC = {'stationary', 'sweep'}
STOP = 24


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def argmax(values):
    # Same first-index tie convention as numpy / the original classifier.
    return max(range(len(values)), key=values.__getitem__)


def label_for_scene(scene):
    if not 1 <= scene <= 150:
        raise ValueError(f'Invalid scene {scene}')
    return CLASSES.index(BLOCK_CLASSES[(scene - 1) // 30])


def predict_readouts(positions, probabilities, actions):
    """Return label-free readouts; ignore zero vectors, deduplicate static views.

    STOP writes the same observation again without acquiring new evidence.
    mean_visits drops only that terminal duplicate; mean_unique also drops
    actual revisits. All-invisible histories produce Unknown (-1).
    """
    if len(positions) != len(probabilities) or len(actions) + 1 != len(positions):
        raise ValueError('Inconsistent trajectory lengths')
    end = len(positions) - int(bool(actions) and actions[-1] == STOP)
    if end < len(positions):
        if positions[-1] != positions[-2] or probabilities[-1] != probabilities[-2]:
            raise ValueError('STOP did not repeat the previous observation')
    seen = {}
    unique, visits = [], []
    for i in range(end):
        p = probabilities[i]
        if len(p) != 5 or any(not math.isfinite(x) or x < 0 for x in p):
            raise ValueError('Invalid class probabilities')
        total = sum(p)
        if total and abs(total - 1) > 2e-4:
            raise ValueError('Visible probability vector does not sum to one')
        pos = tuple(positions[i])
        if pos in seen and probabilities[seen[pos]] != p:
            raise ValueError('Static repeated viewpoint changed its probabilities')
        if total:
            visits.append(i)
            if pos not in seen:
                unique.append(i)
        seen.setdefault(pos, i)
    def mean_prediction(indices):
        if not indices:
            return -1
        return argmax([sum(probabilities[i][c] for i in indices) for c in range(5)])
    best = max(unique, key=lambda i: max(probabilities[i])) if unique else None
    predictions = {
        'final': argmax(probabilities[-1]) if sum(probabilities[-1]) else -1,
        'max_confidence': argmax(probabilities[best]) if best is not None else -1,
        'mean_unique': mean_prediction(unique),
        'mean_visits': mean_prediction(visits),
    }
    return predictions, {
        'unique_visible_indices': unique,
        'best_index': best,
        'unique_positions': len(seen),
        'actual_observations': end,
        'revisits': end - len(seen),
        'stop_duplicate_removed': int(end < len(positions)),
    }


def numpy_reference(t):
    """Independent array implementation, used on stratified source samples."""
    import numpy as np
    p = np.array(t['probabilities'], dtype=np.float64)
    end = len(p) - int(t['actions'][-1] == STOP)
    positions = np.array(t['positions'][:end])
    _, ids = np.unique(positions, axis=0, return_index=True)
    ids = np.sort(ids)
    ids = ids[p[ids].sum(1) > 0]
    visits = p[:end][p[:end].sum(1) > 0]
    unique = p[ids]
    return {
        'final': int(p[-1].argmax()) if p[-1].sum() else -1,
        'max_confidence': int(unique[unique.max(1).argmax()].argmax()) if len(unique) else -1,
        'mean_unique': int(unique.sum(0).argmax()) if len(unique) else -1,
        'mean_visits': int(visits.sum(0).argmax()) if len(visits) else -1,
    }


def score(t, pred, info):
    truth = label_for_scene(t['scene'])
    correct = {m: pred[m] == truth for m in METHODS}
    if correct['final'] != t['final_correct']:
        raise ValueError('Final readout does not match saved result')
    raw = [argmax(p) if sum(p) else -1 for p in t['probabilities']]
    if (raw[0] == truth) != t['initial_correct']:
        raise ValueError('Initial readout does not match saved result')
    if [CLASSES[x] if x >= 0 else 'Unknown' for x in raw] != t['predictions']:
        raise ValueError('Probability class order does not match saved predictions')
    oracle = any(raw[i] == truth for i in info['unique_visible_indices'])
    final = correct['final']
    c = Counter(n=1, oracle=int(oracle), lost_correct=int(oracle and not final),
                never_correct=int(not oracle), all_unknown=int(not info['unique_visible_indices']),
                final_wrong_high_conf=int(not final and max(t['probabilities'][-1]) >= .9),
                max_conf_wrong_high_conf=int(not correct['max_confidence'] and info['best_index'] is not None
                                            and max(t['probabilities'][info['best_index']]) >= .9),
                unique_visible=len(info['unique_visible_indices']), unique_positions=info['unique_positions'],
                actual_observations=info['actual_observations'], revisits=info['revisits'],
                stop_duplicates=info['stop_duplicate_removed'], distance_m=t['distance_m'])
    for m in METHODS:
        c[f'{m}_correct'] = int(correct[m])
        c[f'{m}_rescue'] = int(correct[m] and not final)
        c[f'{m}_harm'] = int(final and not correct[m])
        c[f'{m}_correct_without_correct_view'] = int(correct[m] and not oracle)
        c[f'{m}_missed_oracle'] = int(oracle and not correct[m])
    return c


def stats(values, interval=True):
    values = [float(v) for v in values]
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    # Predeclared five-seed experiment, or a single-seed smoke run.
    ci = [mean - 2.776445105 * sd / math.sqrt(n), mean + 2.776445105 * sd / math.sqrt(n)] if n == 5 and interval else None
    return {'mean': mean, 'sd': sd, 'ci95_t': ci, 'n': n, 'values': values}


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['scene'] == 'all':
            groups[(r['split'], r['mode'], r['policy'], r['subset'])].append(r)
    summaries = []
    for (split, mode, policy, subset), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r['seed'])
        if policy in DETERMINISTIC:
            numeric = [k for k in rs[0] if k not in ('seed', 'split', 'mode', 'policy', 'scene', 'subset')]
            if any(any(r[k] != rs[0][k] for k in numeric) for r in rs[1:]):
                raise ValueError('Deterministic baseline changed across seeds')
            used = rs[:1]
        else:
            used = rs
        if any(r['n'] == 0 for r in used):
            continue
        entry = {'split': split, 'mode': mode, 'policy': policy, 'subset': subset,
                 'seeds': [r['seed'] for r in used], 'episodes_per_seed': [r['n'] for r in used],
                 'randomness': 'deterministic' if policy in DETERMINISTIC else
                               'evaluation_seed' if policy == 'random' else 'policy_training_seed',
                 'methods': {}}
        for m in METHODS:
            entry['methods'][m] = {
                'accuracy': stats([r[f'{m}_correct'] / r['n'] for r in used]),
                'delta_vs_final_pp': stats([100 * (r[f'{m}_correct'] - r['final_correct']) / r['n'] for r in used]),
                'rescued_pp': stats([100 * r[f'{m}_rescue'] / r['n'] for r in used]),
                'harmed_pp': stats([100 * r[f'{m}_harm'] / r['n'] for r in used]),
                'correct_without_correct_view_pp': stats([100 * r[f'{m}_correct_without_correct_view'] / r['n'] for r in used]),
                'missed_oracle_pp': stats([100 * r[f'{m}_missed_oracle'] / r['n'] for r in used]),
            }
        for key in ('oracle', 'lost_correct', 'never_correct', 'all_unknown', 'final_wrong_high_conf', 'max_conf_wrong_high_conf'):
            entry[key + '_rate'] = stats([r[key] / r['n'] for r in used])
        for key in ('unique_visible', 'unique_positions', 'actual_observations', 'revisits', 'distance_m'):
            entry[key + '_mean'] = stats([r[key] / r['n'] for r in used])
        # Fraction of final errors recoverable by selecting an already-correct view.
        entry['lost_fraction_of_final_errors'] = stats([
            r['lost_correct'] / (r['n'] - r['final_correct']) if r['n'] > r['final_correct'] else 0 for r in used])
        summaries.append(entry)
    return summaries


def run(source, out, seeds):
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    aggregate = defaultdict(Counter)
    examples, input_hashes, seed_checks = {}, {}, []
    for seed in seeds:
        directory = source / f'seed_{seed}' / 'evaluation'
        verification = json.loads((directory / 'verification.json').read_text(encoding='utf8'))
        if not verification['passed']:
            raise ValueError(f'Source verification failed: {directory}')
        for name in ('results.json', 'trajectories.jsonl.gz', 'evaluation.csv'):
            p = directory / name
            actual = sha256(p)
            if actual != verification['input_sha256'][str(p.resolve())]:
                raise ValueError(f'Previously verified source changed: {p}')
            input_hashes[str(p.resolve())] = actual
        input_hashes[str((directory / 'verification.json').resolve())] = sha256(directory / 'verification.json')
        splits = json.loads((directory / 'splits.json').read_text(encoding='utf8'))
        input_hashes[str((directory / 'splits.json').resolve())] = sha256(directory / 'splits.json')
        results = json.loads((directory / 'results.json').read_text(encoding='utf8'))
        if tuple(results['classes']) != CLASSES:
            raise ValueError('Unexpected classifier class order')
        starts = defaultdict(set)
        sample_checks = scanned = selected = 0
        with gzip.open(directory / 'trajectories.jsonl.gz', 'rt', encoding='utf8') as f:
            for line in f:
                scanned += 1
                # The source format is fixed and verified above. Train is not a diagnostic test split.
                if '"split":"train"' in line:
                    continue
                t = json.loads(line)
                if t['split'] not in ('simple', 'hard'):
                    continue
                selected += 1
                split, mode, policy, scene = t['split'], t['mode'], t['policy'], t['scene']
                if scene not in splits[split] or policy not in POLICIES or mode not in ('native', 'fixed_moves'):
                    raise ValueError('Unexpected evaluation identity')
                key = (split, mode, policy, scene)
                start = tuple(t[k] for k in ('start_x', 'start_y', 'start_z'))
                if start != tuple(t['positions'][0]) or start in starts[key]:
                    raise ValueError('Duplicated or inconsistent start')
                starts[key].add(start)
                pred, info = predict_readouts(t['positions'], t['probabilities'], t['actions'])
                if (len(starts[key]) - 1) % 100 == 0:
                    if pred != numpy_reference(t):
                        raise ValueError('Independent NumPy implementation disagrees')
                    sample_checks += 1
                c = score(t, pred, info)
                subsets = ['all']
                if info['unique_positions'] > 1:
                    subsets.append('multiple_positions')
                if len(info['unique_visible_indices']) > 1:
                    subsets.append('multiple_visible_views')
                if len({argmax(t['probabilities'][i]) for i in info['unique_visible_indices']}) > 1:
                    subsets.append('disagreeing_views')
                for subset in subsets:
                    aggregate[(seed, split, mode, policy, 'all', subset)].update(c)
                aggregate[(seed, split, mode, policy, scene, 'all')].update(c)
                if seed == seeds[0] and split == 'hard' and mode == 'native' and policy == 'ibe_map':
                    flags = {'mean_rescue': c['mean_unique_rescue'], 'mean_harm': c['mean_unique_harm'],
                             'max_conf_rescue': c['max_confidence_rescue'], 'max_conf_harm': c['max_confidence_harm'],
                             'lost_correct': c['lost_correct'], 'fusion_without_correct_view': c['mean_unique_correct_without_correct_view']}
                    for category, flag in flags.items():
                        if flag and category not in examples:
                            examples[category] = {'note': 'First matching episode in source order; post-selected illustration.',
                                                  'truth': CLASSES[label_for_scene(scene)],
                                                  'readouts': {m: CLASSES[x] if x >= 0 else 'Unknown' for m, x in pred.items()},
                                                  'preprocessing': info, 'trajectory': t}
        expected_grid = {(x, y, z) for x in range(-50, 51, 10) for y in range(-50, 51, 10) for z in range(-100, -59, 10)}
        for key, positions in starts.items():
            if positions != expected_grid:
                raise ValueError(f'Incomplete start coverage: {key}')
        checked_metrics = 0
        for m in results['metrics']:
            if m['split'] == 'train':
                continue
            split = 'hard' if m['split'] == 'hard_fixed_3_moves' else m['split']
            mode = 'fixed_moves' if m['split'] == 'hard_fixed_3_moves' else 'native'
            c = aggregate[(seed, split, mode, m['policy'], 'all', 'all')]
            if c['n'] != m['episodes'] or not math.isclose(c['final_correct'] / c['n'], m['accuracy'], abs_tol=1e-12):
                raise ValueError('Final accuracy did not reproduce original metrics')
            if not math.isclose(c['distance_m'] / c['n'], m['distance_m'], abs_tol=1e-9):
                raise ValueError('Trajectory distance did not reproduce original metrics')
            checked_metrics += 1
        if selected != 302500 or scanned != 726000 or checked_metrics != 20 or len(starts) != 500:
            raise ValueError('Incomplete standard seven-policy evaluation')
        check = {'seed': seed, 'scanned': scanned, 'analyzed': selected, 'reproduced_metric_groups': checked_metrics,
                 'independent_numpy_episode_checks': sample_checks, 'complete_scene_start_groups': len(starts)}
        seed_checks.append(check)
        print(json.dumps(check), flush=True)
    rows = []
    for (seed, split, mode, policy, scene, subset), c in aggregate.items():
        # Counter stores explicit zeros from score, preserving a rectangular CSV.
        rows.append(dict(seed=seed, split=split, mode=mode, policy=policy, scene=scene, subset=subset, **c))
    rows.sort(key=lambda r: (r['seed'], r['split'], r['mode'], r['policy'], str(r['scene']), r['subset']))
    summary = {'schema_version': 1, 'seeds': seeds, 'source': str(source.resolve()),
               'protocol': {
                   'splits': ['simple', 'hard'], 'modes': ['native', 'fixed_moves (hard only)'],
                   'methods': list(METHODS), 'truth_used': 'Scoring and oracle only; no fitted parameters.',
                   'unknown': 'All-zero vectors contribute no evidence; no visible view => Unknown, counted incorrect.',
                   'duplicates': 'Remove terminal STOP duplicate for every method. mean_unique also removes actual viewpoint revisits.',
                   'ties': 'Earliest visible view for maximum-confidence ties; first class index for score ties.',
                   'oracle': 'Any observed visible view has correct top-1; label-assisted view-selection reference, not fusion upper bound.',
                   'interval': 'Paired readout-minus-final differences across five policy seeds; t(4), no multiple-comparison correction.',
                   'deterministic': 'stationary/sweep counted once; five identical copies checked.',
                   'subsets': 'Overlapping descriptive subsets, not additive; all is the main comparison.',
                   'scope': 'Fixed paths, same distance, unchanged stopping. No retraining or evaluation of a new active policy.',
                   'test_usage': 'Retrospective exploratory diagnostic on previously inspected tests; no claim of untouched-test confirmation.',
               }, 'summary': summarize(rows), 'verification': {'passed': True, 'per_seed': seed_checks},
               'source_sha256': input_hashes, 'analysis_sha256': sha256(Path(__file__)),
               'elapsed_seconds': time.time() - started}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    (out / 'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf8')
    with (out / 'counts.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Complete: {out.resolve()} ({time.time() - started:.1f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'output/ibe_multiseed')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/evidence_fusion')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) not in (1, 5):
        parser.error('Use one seed for smoke verification or five distinct seeds for the experiment.')
    run(args.source.resolve(), args.out.resolve(), args.seeds)


if __name__ == '__main__':
    main()
