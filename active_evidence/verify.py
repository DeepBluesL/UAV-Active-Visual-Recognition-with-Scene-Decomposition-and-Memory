"""Independently audit saved active-evidence rollouts using cached observations.

This module imports neither the training implementation nor its rollout/model
functions. It rebuilds the lattice, evidence rules, labels and aggregate scores.
Gate decisions are checked as recorded choices, not rerun through the network.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]
BUDGETS = [0, 10, 20, 40, 60]
CONTROLLERS = ['stationary', 'sweep', 'random', 'ibe_reference', 'gain_final', 'gain_gate']
READOUTS = ['final', 'max_confidence', 'mean', 'gate']
TRAIN = [base + index for base in (0, 30, 60, 90, 120) for index in range(1, 21)]
VALIDATION = [base + index for base in (0, 30, 60, 90, 120) for index in (19, 20)]
DEVELOPMENT = [scene for scene in TRAIN if scene not in VALIDATION]
SPLITS = {
    'simple': [base + index for base in (0, 30, 60, 90, 120) for index in range(21, 26)],
    'hard': [base + index for base in (0, 30, 60, 90, 120) for index in range(26, 31)],
}
GRID = np.asarray(list(itertools.product(range(-50, 51, 10), range(-50, 51, 10),
                                          range(-100, -59, 10))), dtype=np.int64)
COSTS = np.tile(np.asarray([10, 20, 30, 40], dtype=np.int64), 6)
DELTAS = np.zeros((24, 3), dtype=np.int64)
for action in range(24):
    DELTAS[action, action // 8] = COSTS[action] * (1 if action // 4 % 2 == 0 else -1)
POSITION_INDEX = {tuple(position): i for i, position in enumerate(GRID)}
NEXT = np.asarray([[POSITION_INDEX.get(tuple(position + delta), -1) for delta in DELTAS]
                   for position in GRID], dtype=np.int64)
LEGAL = NEXT >= 0


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def close(actual, expected, message, tolerance=1e-7):
    require(np.allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=False), message)


def classify(probabilities):
    predictions = probabilities.argmax(axis=-1)
    return np.where(probabilities.sum(axis=-1) > 0, predictions, -1)


def observation_signature(metadata):
    signature = metadata['signature']
    return {**{key: signature[key] for key in ('pretrained_sha256', 'classifier_sha256', 'preprocess')},
            'dataset_path': signature['source']['dataset_path']}


def cache_fingerprint(metadata_bytes, scenes):
    digest = hashlib.sha256()
    for scene in sorted(scenes):
        digest.update(metadata_bytes[scene])
    return digest.hexdigest()


def load_and_verify_caches(raw_directory, sam_directory):
    """Authenticate both caches, retaining only independently needed raw arrays."""
    probabilities = []
    meta_bytes = {'raw': {}, 'sam': {}}
    signatures = {}
    for scene in range(1, 151):
        observations = {}
        for kind, directory in [('raw', raw_directory), ('sam', sam_directory)]:
            path = directory / 'scenes' / f'{scene:03d}.npz'
            encoded = path.with_suffix('.json').read_bytes()
            metadata = json.loads(encoded)
            require(sha256(path) == metadata['npz_sha256'], f'Cache SHA256 mismatch: {path}')
            signature = observation_signature(metadata)
            if kind in signatures:
                require(signature == signatures[kind], f'Mixed {kind} cache signatures')
            else:
                signatures[kind] = signature
            meta_bytes[kind][scene] = encoded
            with np.load(path, allow_pickle=False) as source:
                observations[kind] = {key: source[key].copy() for key in ('probs', 'visible', 'boxes', 'names')}
                require(np.array_equal(source['positions'], GRID), f'Grid mismatch: {path}')
            values = observations[kind]
            require(values['probs'].shape == (605, 5), f'Probability shape: {path}')
            require(values['visible'].shape == (605,), f'Visibility shape: {path}')
            require(values['boxes'].shape == (605, 4), f'Box shape: {path}')
            require(values['names'].shape == (605,), f'Name shape: {path}')
            p, visible = values['probs'], values['visible'].astype(bool)
            require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), f'Invalid probabilities: {path}')
            close(p[visible].sum(axis=1), np.ones(int(visible.sum())), f'Unnormalized probabilities: {path}', 1e-5)
            require(not np.any(p[~visible]), f'Invisible observation has evidence: {path}')
        for key in observations['raw']:
            require(np.array_equal(observations['raw'][key], observations['sam'][key]),
                    f'Raw/SAM observation mismatch, scene {scene}: {key}')
        probabilities.append(observations['raw']['probs'])
    return np.concatenate(probabilities), meta_bytes, signatures


def verify_selection(selection, config):
    require(selection['config'] == config, 'Selection/result configuration mismatch')
    require(config['seeds'] == SEEDS and config['budgets'] == BUDGETS, 'Unexpected experiment seed/budget protocol')
    require(config['max_moves'] == 5, 'Unexpected movement horizon')
    require(selection['training_scenes'] == DEVELOPMENT, 'Development scenes differ from the 90-scene split')
    require(selection['validation_scenes'] == VALIDATION, 'Validation scenes differ from the 10-scene split')
    require(not set(DEVELOPMENT).intersection(VALIDATION), 'Training/validation overlap')
    require(selection['test_data_loaded'] is False, 'Selection is not declared prior to loading benchmark data')
    code = {name: sha256(ROOT / 'active_evidence' / name) for name in ('experiment.py', 'model.py')}
    require(selection['code'] == code, 'Source changed after selection')
    costs = config['coefficients']
    expected = set(itertools.product(SEEDS, ('gain_final', 'gain_gate'), costs, BUDGETS[1:]))
    seen = set()
    for row in selection['rows']:
        key = row['seed'], row['controller'], row['coefficient'], row['budget_m']
        require(key in expected and key not in seen, f'Invalid/duplicate validation row: {key}')
        seen.add(key)
        require(0 <= row['accuracy'] <= 1 and 0 <= row['distance_m'] <= row['budget_m'], 'Invalid validation metrics')
        close(row['utility'], row['accuracy'] - config['selection_cost_per_m'] * row['distance_m'], 'Validation utility mismatch')
    require(seen == expected, 'Missing validation combinations')
    candidates = {(row['controller'], row['coefficient']): row for row in selection['candidates']}
    require(len(candidates) == len(costs) * 2 == len(selection['candidates']), 'Invalid candidate summary count')
    for controller in ('gain_final', 'gain_gate'):
        summaries = []
        for cost in costs:
            values = [row for row in selection['rows'] if row['controller'] == controller and row['coefficient'] == cost]
            summary = {'coefficient': cost}
            for field in ('utility', 'accuracy', 'distance_m'):
                summary[field] = math.fsum(row[field] for row in values) / len(values)
                close(candidates[controller, cost][field], summary[field], f'Selection aggregate mismatch: {field}')
            summaries.append(summary)
        chosen = max(summaries, key=lambda row: (row['utility'], -row['distance_m'], row['coefficient']))
        require(selection['selected'][controller] == chosen['coefficient'], f'Wrong coefficient selected: {controller}')
    return code


def verify_training(out, config, code, metadata, signatures):
    verified = []
    for phase, scenes, loaded_scenes in [('development', DEVELOPMENT, TRAIN), ('final', TRAIN, list(range(1, 151)))]:
        for seed in SEEDS:
            directory = out / phase / f'seed_{seed}'
            path = directory / 'training.json'
            manifest = read_json(path)
            identity = manifest['identity']
            require(identity['config'] == config and identity['code'] == code, f'Training config/source mismatch: {path}')
            require(identity['training_scenes'] == scenes and identity['seed'] == seed, f'Training split/seed mismatch: {path}')
            expected_cache = {kind: cache_fingerprint(metadata[kind], loaded_scenes) for kind in ('raw', 'sam')}
            expected_cache['observation'] = signatures['raw']
            require(identity['cache'] == expected_cache, f'Training cache identity mismatch: {path}')
            for name in ('gate', 'gain_final', 'gain_gate'):
                checkpoint = directory / f'{name}.pt'
                require(sha256(checkpoint) == manifest['checkpoint_sha256'][name], f'Checkpoint hash mismatch: {checkpoint}')
                require(len(manifest['loss'][name]) == config['epochs'], f'Unexpected completed epochs: {path}, {name}')
                require(all(np.isfinite(row['bce']) for row in manifest['loss'][name]), f'Non-finite training history: {path}')
            verified.append({'path': str(path.relative_to(out)), 'sha256': sha256(path)})
    return verified


def verify_rollout(result, scene_ids, budget, probabilities, context='rollout'):
    """Replay recorded actions/readouts independently, returning audited arrays."""
    starts_expected = np.concatenate([(scene - 1) * 605 + np.arange(605) for scene in scene_ids])
    n = len(starts_expected)
    shapes = {
        'starts': (n,), 'history': (n, 6), 'actions': (n, 5), 'predictions': (n, 4),
        'final_probabilities': (n, 4, 5), 'distance_m': (n,), 'moves': (n,),
        'stop_reason': (n,), 'gate_operations': (n, 3), 'gate_choices': (n, 5), 'labels': (n,),
    }
    require(set(shapes).issubset(result), f'Missing rollout arrays: {context}: {set(shapes) - set(result)}')
    for name, shape in shapes.items():
        require(result[name].shape == shape, f'Shape mismatch {context}: {name}')
        require(np.isfinite(result[name]).all(), f'Non-finite rollout array {context}: {name}')
        if name not in ('final_probabilities', 'distance_m'):
            require(np.issubdtype(result[name].dtype, np.integer), f'Expected integer array {context}: {name}')
    require(np.array_equal(result['starts'], starts_expected), f'Expected 605 common ordered starts per scene: {context}')
    require(np.array_equal(result['history'][:, 0], starts_expected), f'History/start mismatch: {context}')
    class_by_block = np.asarray([0, 1, 3, 2, 4])  # Source scene blocks differ from classifier Sedan/Sportscar order.
    labels = class_by_block[(starts_expected // 605) // 30]
    require(np.array_equal(result['labels'], labels), f'Scene/class mapping mismatch: {context}')
    moves, history, actions = result['moves'], result['history'], result['actions']
    require(((moves >= 0) & (moves <= 5)).all(), f'Invalid move count: {context}')
    present = np.arange(5)[None] < moves[:, None]
    require(np.array_equal(actions >= 0, present), f'Actions are not a contiguous prefix: {context}')
    require(np.all(actions[~present] == -1), f'Action padding must be -1: {context}')
    require(np.array_equal(history >= 0, np.arange(6)[None] <= moves[:, None]), f'History prefix/padding mismatch: {context}')
    require(np.all(history[history < 0] == -1), f'History padding must be -1: {context}')
    require(np.array_equal(result['gate_choices'] >= 0, present), f'Gate choices do not match movement prefix: {context}')
    require(np.all(result['gate_choices'][~present] == -1), f'Gate padding must be -1: {context}')
    memory = probabilities[starts_expected].copy()
    best = memory.copy()
    summed = memory.copy()
    count = (memory.sum(axis=1) > 0).astype(np.float32)
    distance = np.zeros(n, dtype=np.int64)
    operations = np.zeros((n, 3), dtype=np.int64)
    for step in range(5):
        rows = np.flatnonzero(moves > step)
        if not len(rows):
            continue
        current, following = history[rows, step], history[rows, step + 1]
        action = actions[rows, step]
        require(((action >= 0) & (action < 24)).all(), f'Invalid movement action: {context}, step {step}')
        require(((current >= 0) & (current < len(probabilities))).all(), f'Current view outside cache: {context}')
        require(LEGAL[current % 605, action].all(), f'Boundary-invalid movement: {context}, step {step}')
        next_expected = current // 605 * 605 + NEXT[current % 605, action]
        require(np.array_equal(following, next_expected), f'Action does not reach recorded view: {context}, step {step}')
        distance[rows] += COSTS[action]
        require((distance[rows] <= budget).all(), f'Exceeded physical distance budget: {context}, step {step}')
        repeated = (history[rows, :step + 1] == following[:, None]).any(axis=1)
        p = probabilities[following]
        visible = p.sum(axis=1) > 0
        novel = ~repeated & visible
        update_best = novel & (p.max(axis=1) > best[rows].max(axis=1))
        best[rows[update_best]] = p[update_best]
        summed[rows] += p * novel[:, None]
        count[rows] += novel
        op = result['gate_choices'][rows, step]
        require(((op >= 0) & (op < 3)).all(), f'Invalid gate choice: {context}, step {step}')
        require((op[repeated | ~visible] == 1).all(), f'Duplicate/invisible evidence not retained: {context}, step {step}')
        first_visible = (memory[rows].sum(axis=1) == 0) & visible & ~repeated
        require((op[first_visible] == 0).all(), f'First visible observation not accepted: {context}, step {step}')
        proposed = np.stack((p, memory[rows], (memory[rows] + p) / 2), axis=1)
        memory[rows] = proposed[np.arange(len(rows)), op]
        operations[rows, op] += 1
    require(np.array_equal(distance, result['distance_m']), f'Recorded physical distance mismatch: {context}')
    require(np.array_equal(operations, result['gate_operations']), f'Gate operation count mismatch: {context}')
    terminal = history[np.arange(n), moves]
    final = probabilities[terminal]
    mean = summed / np.maximum(count, 1)[:, None]
    expected_p = np.stack((final, best, mean, memory), axis=1)
    close(result['final_probabilities'], expected_p, f'Evidence replay mismatch: {context}', 1e-6)
    expected_prediction = classify(expected_p)
    require(np.array_equal(result['predictions'], expected_prediction), f'Unknown/readout prediction mismatch: {context}')
    # Every lattice point has a legal 10m move; build the minimum independently
    # rather than assume this when deciding whether a budget stop was required.
    cheapest = np.where(LEGAL[terminal % 605], COSTS[None], np.inf).min(axis=1)
    expected_reason = np.where(moves == 5, 3, np.where(budget - distance < cheapest, 2, 1))
    require(np.array_equal(result['stop_reason'], expected_reason), f'Stop reason inconsistent with budget/horizon: {context}')
    return {'labels': labels, 'predictions': expected_prediction, 'distance_m': distance, 'moves': moves,
            'stop_reason': expected_reason, 'gate_operations': operations}


def verify_metrics(rows, artifact, result):
    scenes = SPLITS[artifact['split']]
    checked = 0
    for subset in ['all', *scenes]:
        select = slice(None) if subset == 'all' else slice(scenes.index(subset) * 605, (scenes.index(subset) + 1) * 605)
        labels, prediction = result['labels'][select], result['predictions'][select]
        base = prediction[:, 0] == labels
        for index, readout in enumerate(READOUTS):
            key = artifact['seed'], artifact['split'], artifact['controller'], artifact['budget_m'], subset, readout
            require(key in rows, f'Missing aggregate row: {key}')
            row = rows.pop(key)
            correct = prediction[:, index] == labels
            expected = {
                'accuracy': correct.mean(), 'distance_m': result['distance_m'][select].mean(),
                'moves': result['moves'][select].mean(), 'unknown_rate': (prediction[:, index] < 0).mean(),
                'rescue_rate': (correct & ~base).mean(), 'harm_rate': (~correct & base).mean(),
                'policy_stop_rate': (result['stop_reason'][select] == 1).mean(),
            }
            require(row['episodes'] == len(labels), f'Episode count mismatch: {key}')
            for field, value in expected.items():
                # Saved distance means use float32; independently summing the
                # integer metre costs in float64 can differ by a few ULPs.
                tolerance = 1e-5 if field == 'distance_m' else 1e-7
                close(row[field], value, f'Aggregate {field} mismatch: {key}', tolerance)
            require(row['gate_operation_counts'] == result['gate_operations'][select].sum(axis=0).tolist(), f'Aggregate gate count mismatch: {key}')
            close(row['accuracy'] - base.mean(), row['rescue_rate'] - row['harm_rate'], f'Rescue/harm identity failed: {key}')
            checked += 1
    return checked


def verify(out):
    result_path, selection_path = out / 'results.json', out / 'selection.json'
    document, selection = read_json(result_path), read_json(selection_path)
    require(document['complete'] is True, 'Final experiment is not complete')
    require(document['selection_sha256'] == sha256(selection_path), 'Selection artifact changed')
    code = verify_selection(selection, document['config'])
    probabilities, metadata, signatures = load_and_verify_caches(ROOT / 'output/map/cache', ROOT / 'output/ibe_map/cache')
    expected_cache = {kind: cache_fingerprint(metadata[kind], range(1, 151)) for kind in ('raw', 'sam')}
    expected_cache['observation'] = signatures['raw']
    require(document['cache'] == expected_cache, 'Result cache fingerprints mismatch')
    training_manifests = verify_training(out, document['config'], code, metadata, signatures)
    expected_artifacts = set(itertools.product(SEEDS, SPLITS, CONTROLLERS, BUDGETS))
    seen_artifacts, seen_paths = set(), set()
    metric_rows = {}
    for row in document['rows']:
        key = tuple(row[name] for name in ('seed', 'split', 'controller', 'budget_m', 'scene', 'readout'))
        require(key not in metric_rows, f'Duplicate aggregate row: {key}')
        metric_rows[key] = row
    checked_rows = episodes = 0
    reference_hashes = {}
    for artifact in document['artifacts']:
        key = tuple(artifact[name] for name in ('seed', 'split', 'controller', 'budget_m'))
        require(key in expected_artifacts and key not in seen_artifacts, f'Unexpected/duplicate rollout artifact: {key}')
        seen_artifacts.add(key)
        path = (out / artifact['path']).resolve()
        require(path.is_relative_to(out.resolve()) and path not in seen_paths, f'Invalid/duplicate artifact path: {path}')
        seen_paths.add(path)
        require(sha256(path) == artifact['sha256'], f'Rollout hash mismatch: {path}')
        if artifact['controller'] == 'ibe_reference':
            seed = artifact['seed']
            if seed not in reference_hashes:
                checkpoint = ROOT / 'output/ibe_multiseed' / f'seed_{seed}' / 'ibe_map/train/last.pt'
                reference_hashes[seed] = sha256(checkpoint)
                reference_config = read_json(checkpoint.parent / 'training.json')['config']
                require(reference_config['variant'] == 'ibe_map' and reference_config['seed'] == seed,
                        f'Reference variant/seed mismatch: {checkpoint}')
                require(reference_config['training_scenes'] == TRAIN, f'Reference training scenes mismatch: {checkpoint}')
                require(reference_config['observation_signature'] == signatures['sam'], f'Reference SAM signature mismatch: {checkpoint}')
            require(artifact['reference_sha256'] == reference_hashes[seed], f'Reference checkpoint hash mismatch: {path}')
        else:
            require(artifact['reference_sha256'] is None, f'Unexpected reference hash: {path}')
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        audited = verify_rollout(arrays, SPLITS[artifact['split']], artifact['budget_m'], probabilities, path.name)
        if artifact['controller'] == 'stationary':
            require(not np.any(audited['moves']), f'Stationary controller moved: {path}')
        checked_rows += verify_metrics(metric_rows, artifact, audited)
        episodes += len(audited['labels'])
        if len(seen_artifacts) % 25 == 0:
            print(f'Verified {len(seen_artifacts)}/{len(expected_artifacts)} artifacts; {episodes:,} episodes', flush=True)
    require(seen_artifacts == expected_artifacts, 'Missing expected rollout artifacts')
    require(not metric_rows, f'Unmatched aggregate rows remain: {len(metric_rows)}')
    return {
        'passed': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'results_sha256': sha256(result_path), 'selection_sha256': sha256(selection_path),
        'verifier_sha256': sha256(Path(__file__)), 'source_sha256': code,
        'artifact_count': len(seen_artifacts), 'episodes': episodes, 'aggregate_rows': checked_rows,
        'cache_files_verified': 300, 'training_manifests': training_manifests,
        'gate_choice_replay': 'Every recorded step: replace/retain/fuse, visibility and repeated-view overrides.',
        'checks': ['artifact and cache SHA256', '90/10 development split and selected coefficients',
                   'training identities and model checkpoint hashes', '605 identical starts per scene',
                   'independent lattice movement and legality', 'hard physical distance caps',
                   'Unknown and scene/class ordering', 'final/max-confidence/unique-mean readouts',
                   'recorded gate choices and reconstructed evidence', 'aggregate accuracy/distance/moves',
                   'aggregate rescue/harm/Unknown/stop rates', 'reference checkpoint provenance'],
        'scope': 'Recorded actions and gate choices are audited against physical/evidence rules. Neural predictions are not rerun; selection calibration values are not independently re-inferred.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/active_evidence')
    args = parser.parse_args()
    try:
        report = verify(args.out)
    except Exception as error:
        if args.out.is_dir():
            write_json(args.out / 'verification.json', {'passed': False, 'error_type': type(error).__name__, 'error': str(error)})
        raise
    write_json(args.out / 'verification.json', report)
    print(json.dumps({key: report[key] for key in ('passed', 'artifact_count', 'episodes', 'aggregate_rows')}, indent=2))


if __name__ == '__main__':
    main()
