"""Independent, CPU-only integrity checks for a complete IBE-MAP evaluation.

This deliberately does not import the training/evaluation code or torch. It
reconstructs rewards, distances, classifier decisions and aggregates from the
serialized trajectories and the original per-scene probability cache.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ('Hatchback', 'Pickup', 'Sedan', 'Sportscar', 'SUV')
BLOCK_CLASSES = ('Hatchback', 'Pickup', 'Sportscar', 'Sedan', 'SUV')
GRID = [(x, y, z) for x in range(-50, 51, 10) for y in range(-50, 51, 10)
        for z in range(-100, -59, 10)]
LOOKUP = {p: i for i, p in enumerate(GRID)}
STOP = 24
FIELDS = ('split', 'mode', 'policy', 'scene', 'start_x', 'start_y', 'start_z',
          'initial_correct', 'final_correct', 'return', 'distance_m', 'decisions',
          'stop_reason', 'final_confidence', 'success')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, tol=2e-6):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def load_raw(cache_dir, scene):
    p = Path(cache_dir) / 'scenes' / f'{scene:03d}.npz'
    require(p.is_file(), f'Missing raw cache: {p}')
    with np.load(p, allow_pickle=False) as z:
        required = {'positions', 'probs', 'visible', 'names'}
        require(required.issubset(z.files), f'{p} missing arrays: {required-set(z.files)}')
        result = {k: z[k].copy() for k in required}
    require(result['positions'].shape == (605, 3), f'Bad positions shape: {p}')
    require(np.array_equal(result['positions'], np.asarray(GRID)), f'Raw grid order differs: {p}')
    require(result['probs'].shape == (605, 5), f'Bad probs shape: {p}')
    return result


def class_index(scene):
    return CLASSES.index(BLOCK_CLASSES[(scene - 1) // 30])


def scene_location(scene, position):
    p = tuple(int(v) for v in position)
    require(p in LOOKUP, f'Out-of-grid position {p} in scene {scene}')
    return LOOKUP[p]


def validate_trajectory(t, protocol, cache_dir, cache, checks):
    scene = int(t['scene'])
    positions = [tuple(int(v) for v in p) for p in t['positions']]
    actions = [int(a) for a in t['actions']]
    rewards = [float(r) for r in t['rewards']]
    probs = np.asarray(t['probabilities'], dtype=np.float64)
    preds = t['predictions']
    decisions = int(t['decisions'])
    require(decisions >= 1 and decisions <= int(protocol['max_decisions']), f'Bad decisions: {t}')
    require(len(positions) == decisions + 1, f'positions != decisions+1 for {scene}/{t.get("policy")}')
    require(len(actions) == len(rewards) == decisions, f'Action/reward lengths disagree: {scene}/{t.get("policy")}')
    require(probs.shape == (decisions + 1, 5), f'Bad trajectory probability shape: {probs.shape}')
    require(len(preds) == decisions + 1, 'Prediction count must equal observation count')
    require(np.isfinite(probs).all() and (probs >= -1e-7).all(), 'Invalid probabilities')
    raw = cache.get(scene)
    if raw is None:
        raw = load_raw(cache_dir, scene)
        cache[scene] = raw
        checks['raw_scenes_loaded'] += 1
    locs = [scene_location(scene, p) for p in positions]
    for j, (loc, p, pred) in enumerate(zip(locs, probs, preds)):
        require(np.allclose(p, raw['probs'][loc], atol=2e-6, rtol=2e-6),
                f'Probability/cache mismatch scene={scene} position={positions[j]}')
        visible = bool(raw['visible'][loc])
        if visible:
            require(close(p.sum(), 1, tol=2e-4), 'Visible observation probabilities do not sum to one')
        else:
            require(not np.any(p), 'Invisible observation must have zero probabilities')
        predicted = CLASSES[int(np.argmax(raw['probs'][loc]))] if visible else 'Unknown'
        require(pred == predicted, f'Prediction/cache mismatch at scene={scene}, position={positions[j]}')
    truth = class_index(scene)
    initial_loc, final_loc = locs[0], locs[-1]
    initial_correct = bool(raw['visible'][initial_loc] and np.argmax(raw['probs'][initial_loc]) == truth)
    final_correct = bool(raw['visible'][final_loc] and np.argmax(raw['probs'][final_loc]) == truth)
    success = bool(raw['visible'][final_loc] and raw['probs'][final_loc, truth] > protocol['recognition_threshold'])
    if protocol['success_mode'] == 'top1_and_true_probability':
        success = success and final_correct
    require(bool(t['initial_correct']) == initial_correct, 'initial_correct disagrees with cache')
    require(bool(t['final_correct']) == final_correct, 'final_correct disagrees with cache')
    require(bool(t['success']) == success, 'success disagrees with protocol/cache')
    require(close(t['final_confidence'], raw['probs'][final_loc].max()), 'final_confidence disagrees with cache')
    discounted_return = 0.0
    distance = 0.0
    movement_count = 0
    for i, (a, reward) in enumerate(zip(actions, rewards)):
        require(0 <= a <= STOP, f'Action outside 0..24: {a}')
        before, after = positions[i], positions[i + 1]
        if a == STOP:
            require(before == after, f'STOP moved position {before} -> {after}')
            require(i == decisions - 1, 'STOP must be terminal')
            units = 0
        else:
            # Action layout is direction-major, four ranges for each +/- axis.
            direction, unit = divmod(a, 4)
            axis, sign = divmod(direction, 2)
            delta = [0, 0, 0]
            delta[axis] = (1 if sign == 0 else -1) * (unit + 1) * 10
            expected = tuple(before[k] + delta[k] for k in range(3))
            require(after == expected, f'Action {a} does not match {before}->{after}; expected {expected}')
            require(after in LOOKUP, f'Illegal boundary move {before}->{after}')
            units = unit + 1
            movement_count += 1
        distance += units * 10
        terminal = i == decisions - 1
        expected_reward = ((protocol['terminal_reward'] if success else -protocol['terminal_reward'])
                           if terminal else -protocol['step_penalty'])
        expected_reward -= protocol['distance_penalty_per_unit'] * units
        require(close(reward, expected_reward), f'Reward mismatch scene={scene}, step={i}: {reward} != {expected_reward}')
        discounted_return += (protocol['gamma'] ** i) * expected_reward
    require(close(t['return'], discounted_return), f'Discounted return mismatch: {t["return"]} != {discounted_return}')
    require(close(t['distance_m'], distance), 'Distance does not match actions')
    if actions[-1] != STOP:
        require(decisions == int(protocol['max_decisions']), 'Movement-only episode ended before decision budget')
        reason = 'max_decisions_after_move'
    elif t['mode'] == 'fixed_moves':
        reason = 'fixed_movement_budget'
    elif t['policy'] in ('sweep', 'random'):
        reason = 'confidence_threshold'
        require(float(t['final_confidence']) >= float(protocol['heuristic_stop_threshold']), 'Heuristic STOP below threshold')
    else:
        reason = 'policy_stop'
    require(t['stop_reason'] == reason, 'Stop reason disagrees with actions/policy')
    if t['mode'] == 'fixed_moves':
        require(actions == actions[:3] + [STOP] if len(actions) == 4 else False,
                f'fixed3 must contain exactly 3 moves then STOP, got {actions}')
    if t['mode'] == 'fixed_moves' and len(actions) == 4:
        require(all(a != STOP for a in actions[:3]), 'fixed3 movement prefix contains STOP')
    return {'correct': final_correct, 'initial_correct': initial_correct, 'success': success,
            'returns': discounted_return, 'distance': distance, 'decisions': decisions,
            'stopped': actions[-1] == STOP, 'last_action': actions[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('output/ibe_map'))
    ap.add_argument('--raw-cache', type=Path, default=Path('output/map/cache'))
    ap.add_argument('--data', type=Path, default=Path('dataset'))
    ap.add_argument('--verification-output', type=Path, default=None)
    args = ap.parse_args()
    out = args.out
    result_path, csv_path, trace_path = out/'results.json', out/'evaluation.csv', out/'trajectories.jsonl.gz'
    for p in (result_path, csv_path, trace_path):
        require(p.is_file(), f'Missing required result file: {p}')
    result = json.loads(result_path.read_text(encoding='utf8'))
    protocol = result['protocol']
    metrics = result['metrics']
    require(metrics, 'No metrics in results.json')
    groups = {(m['split'], m['policy']): m for m in metrics}
    require(len(groups) == len(metrics), 'Duplicate split/policy metrics')
    expected_episodes = sum(int(m['episodes']) for m in metrics)
    csv_rows = {}
    csv_keys = set()
    with csv_path.open(newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        require(set(FIELDS).issubset(reader.fieldnames or []), 'evaluation.csv lacks required fields')
        for row in reader:
            key = (row['split'], row['mode'], row['policy'], int(row['scene']), int(row['start_x']), int(row['start_y']), int(row['start_z']))
            require(key not in csv_keys, f'Duplicate CSV episode key: {key}')
            csv_keys.add(key)
            csv_rows[key] = row
    aggregates = defaultdict(list)
    common_starts = defaultdict(set)
    raw_cache = {}
    checks = {'raw_scenes_loaded': 0}
    trace_keys = set()
    trajectory_count = 0
    with gzip.open(trace_path, 'rt', encoding='utf8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            t = json.loads(line)
            trajectory_count += 1
            key = (t['split'], t['mode'], t['policy'], int(t['scene']), *map(int, t['positions'][0]))
            require(key not in trace_keys, f'Duplicate trajectory key line {line_no}: {key}')
            trace_keys.add(key)
            require(key in csv_rows, f'Trajectory has no CSV row: {key}')
            row = csv_rows[key]
            for field in FIELDS:
                v = t[field]
                if field in ('scene', 'decisions'):
                    require(int(row[field]) == int(v), f'CSV/trajectory mismatch {field} at {key}')
                elif field in ('initial_correct', 'final_correct', 'success'):
                    require(row[field].lower() == str(bool(v)).lower(), f'CSV/trajectory mismatch {field} at {key}')
                elif field in ('return', 'distance_m', 'final_confidence'):
                    require(close(row[field], v), f'CSV/trajectory mismatch {field} at {key}')
                else:
                    require(str(row[field]) == str(v), f'CSV/trajectory mismatch {field} at {key}')
            stats = validate_trajectory(t, protocol, args.raw_cache, raw_cache, checks)
            metric_split = t['split'] if t['mode'] == 'native' else 'hard_fixed_3_moves'
            g = (metric_split, t['policy'])
            require(g in groups, f'Trajectory group has no metrics entry: {g}')
            start = (int(t['scene']), *map(int, t['positions'][0]))
            common_starts[g].add(start)
            aggregates[g].append(stats)
    require(csv_keys == trace_keys, f'CSV/trajectory key sets differ: csv={len(csv_keys)}, trace={len(trace_keys)}')
    require(trajectory_count == expected_episodes, f'Expected {expected_episodes} trajectories from metrics, got {trajectory_count}')
    # Policies in each evaluation group must cover exactly the same scene/start tuples.
    by_eval_group = defaultdict(list)
    for split, policy in groups:
        eval_group = 'hard' if split == 'hard_fixed_3_moves' else split
        by_eval_group[eval_group].append((split, policy))
    for eval_group, members in by_eval_group.items():
        sets = [common_starts[g] for g in members]
        require(sets and all(s == sets[0] for s in sets[1:]), f'Policies have unmatched starts in {eval_group}')
        require(len(sets[0]) > 0, f'No starts in {eval_group}')
    for g, metric in groups.items():
        rows = aggregates[g]
        require(len(rows) == int(metric['episodes']), f'Episode count mismatch for {g}')
        starts = common_starts[g]
        require(len(starts) == int(metric['episodes']), f'Duplicate/missing starts for {g}')
        scene_ids = {scene for scene, *_ in starts}
        split_name = g[0]
        if split_name == 'train':
            expected_scenes = {base + offset for base in (0, 30, 60, 90, 120) for offset in range(1, 21)}
        elif split_name == 'simple':
            expected_scenes = {base + offset for base in (0, 30, 60, 90, 120) for offset in range(21, 26)}
        else:
            expected_scenes = {base + offset for base in (0, 30, 60, 90, 120) for offset in range(26, 31)}
        require(scene_ids == expected_scenes, f'{g} does not contain the full expected scene split')
        per_scene = defaultdict(int)
        for scene, *_ in starts:
            per_scene[scene] += 1
        require(all(count == 605 for count in per_scene.values()), f'{g} must contain all 605 starts in every scene')
        n = len(rows)
        avg = lambda key: sum(float(x[key]) for x in rows) / n
        boolmean = lambda key: sum(bool(x[key]) for x in rows) / n
        ints = lambda key: sum(int(x[key]) for x in rows)
        got = {'accuracy': boolmean('correct'), 'initial_accuracy': boolmean('initial_correct'),
               'return_mean': avg('returns'), 'distance_m': avg('distance'), 'decisions': avg('decisions'),
               'success_rate': boolmean('success'), 'stop_rate': boolmean('stopped'),
               'budget_end_rate': 1-boolmean('stopped'),
               'wrong_to_correct': sum(not x['initial_correct'] and x['correct'] for x in rows),
               'correct_to_wrong': sum(x['initial_correct'] and not x['correct'] for x in rows),
               'wrong_stop_count': sum(x['stopped'] and not x['correct'] for x in rows)}
        for name, value in got.items():
            require(name in metric and close(metric[name], value), f'Metric mismatch {g}.{name}: recorded={metric.get(name)} recomputed={value}')
    inputs = [result_path, csv_path, trace_path]
    inputs.extend(sorted((Path(args.raw_cache)/'scenes').glob('*.npz')))
    verification = {'schema_version': 1, 'passed': True,
        'checks': {'csv_trace_row_match': True, 'unique_episode_keys': True, 'matched_policy_starts': True,
                   'trajectory_invariants': True, 'raw_cache_probabilities_and_predictions': True,
                   'protocol_rewards_success_and_aggregates': True},
        'counts': {'metrics_groups': len(groups), 'trajectories': trajectory_count, 'csv_rows': len(csv_rows),
                   'unique_episode_keys': len(trace_keys), 'raw_scenes_loaded': checks['raw_scenes_loaded']},
        'input_sha256': {str(p): sha256(p) for p in inputs},
        'data_path': str(args.data.resolve()), 'raw_cache': str(args.raw_cache.resolve())}
    destination = args.verification_output or (out/'verification.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(verification, indent=2), encoding='utf8')
    print(json.dumps({'passed': True, **verification['counts'], 'verification': str(destination.resolve())}, indent=2))


if __name__ == '__main__':
    main()
