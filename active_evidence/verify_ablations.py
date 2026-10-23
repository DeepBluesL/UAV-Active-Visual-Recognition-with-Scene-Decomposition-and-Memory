"""Independently audit ablation provenance, recorded paths and evidence updates.

No training, model or rollout implementation is imported. Neural decisions need
the separate checkpoint replay; this verifier reconstructs their recorded
consequences and the two deterministic rules from authenticated observations.
"""
from __future__ import annotations

import argparse
import itertools
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from active_evidence import verify as base


ROOT = Path(__file__).resolve().parents[1]
SEEDS, BUDGETS = base.SEEDS, base.BUDGETS
READOUTS = ['final', 'max_confidence', 'mean', 'rule_latest', 'rule_agree',
            'gate2_full', 'gate3_full', 'gate2_noembed', 'gate3_noembed']
GATE_NAMES = READOUTS[3:]
CONTROLLERS = ['gain_full', 'joint_reference', 'gain_noembed', 'gain_noembed_fixed_cost']
MODELS = ['gate3_full', 'gate2_full', 'gate3_noembed', 'gate2_noembed', 'gain_full', 'gain_noembed']
CODE_FILES = ['experiment.py', 'model.py', 'ablation_model.py', 'ablations.py', 'ABLATION_PROTOCOL.md']
require, close, sha256, read_json = base.require, base.close, base.sha256, base.read_json


def confined_path(directory, relative, label):
    require(isinstance(relative, str), f'Missing {label} path')
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), f'{label} path escapes output directory: {relative}')
    return path


def verify_selection(selection, config, source_selection, source_selection_path):
    require(selection['config'] == config == source_selection['config'], 'Changed experiment protocol configuration')
    require(config['seeds'] == SEEDS and config['budgets'] == BUDGETS, 'Changed seed/budget protocol')
    require(config['max_moves'] == 5 and config['epochs'] == 20, 'Changed horizon/epochs')
    require(selection['training_scenes'] == base.DEVELOPMENT, 'Incorrect 90-scene development split')
    require(selection['validation_scenes'] == base.VALIDATION, 'Incorrect 10-scene validation split')
    require(selection['test_data_loaded'] is False, 'Validation selection was not declared before loading test scenes')
    require(selection['source_selection_sha256'] == sha256(source_selection_path), 'Original selection changed')
    code = {name: sha256(ROOT / 'active_evidence' / name) for name in CODE_FILES}
    require(selection['code'] == code, 'Ablation source/protocol changed after selection')
    costs = config['coefficients']
    expected = set(itertools.product(SEEDS, costs, BUDGETS[1:]))
    seen = set()
    for row in selection['rows']:
        key = row['seed'], row['coefficient'], row['budget_m']
        require(key in expected and key not in seen, f'Invalid/duplicate validation row: {key}')
        seen.add(key)
        require(0 <= row['accuracy'] <= 1 and 0 <= row['distance_m'] <= row['budget_m'],
                f'Invalid validation metrics: {key}')
        close(row['utility'], row['accuracy'] - config['selection_cost_per_m'] * row['distance_m'],
              f'Validation utility mismatch: {key}')
    require(seen == expected, 'Missing validation combinations')
    candidates = {row['coefficient']: row for row in selection['candidates']}
    require(set(candidates) == set(costs) and len(candidates) == len(selection['candidates']),
            'Invalid validation candidate summaries')
    summaries = []
    for cost in costs:
        rows = [row for row in selection['rows'] if row['coefficient'] == cost]
        summary = {'coefficient': cost}
        for field in ('utility', 'accuracy', 'distance_m'):
            summary[field] = math.fsum(row[field] for row in rows) / len(rows)
            close(candidates[cost][field], summary[field], f'Validation aggregate mismatch: {cost}, {field}')
        summaries.append(summary)
    selected = max(summaries, key=lambda row: (row['utility'], -row['distance_m'], row['coefficient']))
    expected_selected = {
        'gain_full': source_selection['selected']['gain_final'],
        'gain_noembed': selected['coefficient'],
        'gain_noembed_fixed_cost': source_selection['selected']['gain_final'],
    }
    require(selection['selected'] == expected_selected, 'Incorrect selected/fixed control coefficients')
    return code


def compare_checkpoint(path, source_path):
    """Check reproduced weights directly, without constructing a model."""
    import torch

    current = torch.load(path, weights_only=True, map_location='cpu')
    original = torch.load(source_path, weights_only=True, map_location='cpu')
    require(current.keys() == original.keys(), f'Reproduced checkpoint keys differ: {path}')
    maximum = 0.0
    for name in current:
        require(current[name].shape == original[name].shape, f'Reproduced checkpoint shape differs: {path}, {name}')
        require(torch.isfinite(current[name]).all().item(), f'Nonfinite checkpoint weights: {path}, {name}')
        difference = float((current[name].double() - original[name].double()).abs().max())
        maximum = max(maximum, difference)
    require(maximum <= 1e-6, f'Original checkpoint reproduction failed: {path}, max_abs={maximum}')
    return maximum


def verify_training(out, source, config, code, metadata, signatures):
    verified = []
    for phase, scenes, loaded in [('development', base.DEVELOPMENT, base.TRAIN),
                                   ('final', base.TRAIN, list(range(1, 151)))]:
        for seed in SEEDS:
            directory = out / phase / f'seed_{seed}'
            source_directory = source / phase / f'seed_{seed}'
            path = directory / 'training.json'
            manifest = read_json(path)
            identity = manifest['identity']
            require(identity['config'] == config and identity['code'] == code, f'Training config/code mismatch: {path}')
            require(identity['training_scenes'] == scenes and identity['seed'] == seed, f'Training split/seed mismatch: {path}')
            require(identity['source_training_sha256'] == sha256(source_directory / 'training.json'),
                    f'Original training manifest changed: {path}')
            expected_cache = {kind: base.cache_fingerprint(metadata[kind], loaded) for kind in ('raw', 'sam')}
            expected_cache['observation'] = signatures['raw']
            require(identity['cache'] == expected_cache, f'Training cache identity mismatch: {path}')
            require(set(manifest['checkpoint_sha256']) == set(MODELS), f'Incomplete model checkpoint set: {path}')
            require(set(manifest['loss']) == set(MODELS), f'Incomplete training histories: {path}')
            for name in MODELS:
                checkpoint = directory / f'{name}.pt'
                require(sha256(checkpoint) == manifest['checkpoint_sha256'][name], f'Checkpoint SHA256 mismatch: {checkpoint}')
                history = manifest['loss'][name]
                require(len(history) == config['epochs'], f'Wrong completed epochs: {path}, {name}')
                require([row['epoch'] for row in history] == list(range(1, config['epochs'] + 1)),
                        f'Incorrect epoch sequence: {path}, {name}')
                require(all(np.isfinite(row['bce']) and row['bce'] >= 0 for row in history),
                        f'Invalid training loss: {path}, {name}')
            reproduced = {}
            for name, original_name, field in [('gate3_full', 'gate', 'gate_max_abs'),
                                                ('gain_full', 'gain_final', 'gain_max_abs')]:
                maximum = compare_checkpoint(directory / f'{name}.pt', source_directory / f'{original_name}.pt')
                recorded = manifest['reproduced_reference'][field]
                require(np.isfinite(recorded) and 0 <= recorded <= 1e-6, f'Invalid reproduction result: {path}, {field}')
                close(recorded, maximum, f'Reproduction value mismatch: {path}, {field}', 1e-12)
                reproduced[field] = maximum
            verified.append({'path': str(path.relative_to(out)), 'sha256': sha256(path),
                             'reproduced_reference': reproduced})
    return verified


def verify_rule_choices(arrays, probabilities, context):
    """Derive rules from acquired observations, including mandatory overrides."""
    history, moves = arrays['history'], arrays['moves']
    for gate_index, name in enumerate(GATE_NAMES):
        choices = arrays['gate_choices'][:, gate_index]
        if name.startswith('gate2_'):
            require(not np.any(choices == 2), f'Binary gate selected fusion: {context}, {name}')
        if name not in ('rule_latest', 'rule_agree'):
            continue
        memory = probabilities[arrays['starts']].copy()
        for step in range(5):
            rows = np.flatnonzero(moves > step)
            if not len(rows):
                continue
            following = history[rows, step + 1]
            new = probabilities[following]
            old = memory[rows]
            repeated = (history[rows, :step + 1] == following[:, None]).any(axis=1)
            visible, old_visible = new.sum(axis=1) > 0, old.sum(axis=1) > 0
            if name == 'rule_latest':
                expected = np.zeros(len(rows), dtype=np.int64)
            else:
                agree = new.argmax(axis=1) == old.argmax(axis=1)
                expected = np.where(agree, 2, np.where(new.max(axis=1) >= old.max(axis=1), 0, 1))
            expected = np.where(~old_visible & visible, 0, expected)
            expected = np.where(~visible | repeated, 1, expected)
            require(np.array_equal(choices[rows, step], expected),
                    f'Deterministic rule mismatch: {context}, {name}, step {step}')
            candidates = np.stack((new, old, (old + new) / 2), axis=1)
            memory[rows] = candidates[np.arange(len(rows)), expected]


def verify_rollout(arrays, scenes, budget, probabilities, context):
    n = len(scenes) * 605
    shapes = {'predictions': (n, len(READOUTS)), 'final_probabilities': (n, len(READOUTS), 5),
              'gate_choices': (n, len(GATE_NAMES), 5), 'gate_operations': (n, len(GATE_NAMES), 3)}
    for name, shape in shapes.items():
        require(name in arrays and arrays[name].shape == shape, f'Ablation array shape mismatch: {context}, {name}')
    predictions, operations = [], []
    audited = None
    for gate_index, name in enumerate(GATE_NAMES):
        indices = [0, 1, 2, gate_index + 3]
        adapted = {**arrays, 'predictions': arrays['predictions'][:, indices],
                   'final_probabilities': arrays['final_probabilities'][:, indices],
                   'gate_choices': arrays['gate_choices'][:, gate_index],
                   'gate_operations': arrays['gate_operations'][:, gate_index]}
        replay = base.verify_rollout(adapted, scenes, budget, probabilities, f'{context}/{name}')
        if audited is None:
            audited = replay
            predictions.extend(replay['predictions'][:, index] for index in range(3))
        predictions.append(replay['predictions'][:, 3])
        operations.append(replay['gate_operations'])
    verify_rule_choices(arrays, probabilities, context)
    audited['predictions'] = np.stack(predictions, axis=1)
    audited['gate_operations'] = np.stack(operations, axis=1)
    return audited


def verify_source_rollout(artifact, arrays, source, source_artifacts):
    original_controller = {'gain_full': 'gain_final', 'joint_reference': 'gain_gate'}.get(artifact['controller'])
    if original_controller is None:
        require(artifact['source_path'] is None and artifact['source_sha256'] is None,
                'New controller incorrectly claims an original trajectory')
        return
    key = artifact['seed'], artifact['split'], original_controller, artifact['budget_m']
    require(key in source_artifacts, f'Missing original trajectory metadata: {key}')
    original = source_artifacts[key]
    require(artifact['source_path'] == original['path'] and artifact['source_sha256'] == original['sha256'],
            f'Original trajectory identity mismatch: {key}')
    path = confined_path(source, artifact['source_path'], 'Original rollout')
    require(sha256(path) == artifact['source_sha256'], f'Original trajectory SHA256 mismatch: {path}')
    with np.load(path, allow_pickle=False) as saved:
        for name in ('starts', 'history', 'actions', 'distance_m', 'moves', 'stop_reason', 'labels'):
            require(np.array_equal(arrays[name], saved[name]), f'Original trajectory changed: {key}, {name}')
        for name in ('predictions', 'final_probabilities'):
            require(np.array_equal(arrays[name][:, :3], saved[name][:, :3]),
                    f'Original baseline readout changed: {key}, {name}')


def verify_metrics(rows, artifact, result):
    scenes = base.SPLITS[artifact['split']]
    checked = 0
    for subset in ['all', *scenes]:
        index = slice(None) if subset == 'all' else slice(scenes.index(subset) * 605, (scenes.index(subset) + 1) * 605)
        labels, predictions = result['labels'][index], result['predictions'][index]
        final_correct = predictions[:, 0] == labels
        for readout_index, readout in enumerate(READOUTS):
            key = artifact['seed'], artifact['split'], artifact['controller'], artifact['budget_m'], subset, readout
            require(key in rows, f'Missing aggregate row: {key}')
            row = rows.pop(key)
            correct = predictions[:, readout_index] == labels
            expected = {
                'accuracy': correct.mean(), 'distance_m': result['distance_m'][index].mean(),
                'moves': result['moves'][index].mean(), 'unknown_rate': (predictions[:, readout_index] < 0).mean(),
                'rescue_rate': (correct & ~final_correct).mean(), 'harm_rate': (~correct & final_correct).mean(),
                'policy_stop_rate': (result['stop_reason'][index] == 1).mean(),
            }
            require(row['episodes'] == len(labels), f'Episode count mismatch: {key}')
            for field, value in expected.items():
                close(row[field], value, f'Aggregate {field} mismatch: {key}', 1e-5 if field == 'distance_m' else 1e-7)
            expected_operations = None if readout_index < 3 else result['gate_operations'][index, readout_index - 3].sum(axis=0).tolist()
            require(row['gate_operation_counts'] == expected_operations, f'Aggregate gate counts mismatch: {key}')
            close(row['accuracy'] - final_correct.mean(), row['rescue_rate'] - row['harm_rate'], f'Rescue/harm identity mismatch: {key}')
            checked += 1
    return checked


def verify(out, source):
    out, source = out.resolve(), source.resolve()
    document = read_json(out / 'results.json')
    selection = read_json(out / 'selection.json')
    source_document = read_json(source / 'results.json')
    source_selection = read_json(source / 'selection.json')
    source_verification = read_json(source / 'verification.json')
    require(document['complete'] is True and source_document['complete'] is True, 'Experiment incomplete')
    require(source_verification['passed'] is True, 'Original independent verification did not pass')
    for name in ('results', 'selection'):
        require(source_verification[name + '_sha256'] == sha256(source / (name + '.json')),
                f'Original {name} changed after independent verification')
    require(document['readouts'] == READOUTS and document['gate_names'] == GATE_NAMES and document['controllers'] == CONTROLLERS,
            'Incorrect ablation coverage definitions')
    require(document['selection_sha256'] == sha256(out / 'selection.json'), 'Selection artifact changed')
    require(source_document['selection_sha256'] == sha256(source / 'selection.json'), 'Original selection artifact changed')
    source_code = base.verify_selection(source_selection, source_document['config'])
    code = verify_selection(selection, document['config'], source_selection, source / 'selection.json')
    require(document['code'] == code, 'Result source/protocol identity mismatch')
    probabilities, metadata, signatures = base.load_and_verify_caches(ROOT / 'output/map/cache', ROOT / 'output/ibe_map/cache')
    expected_cache = {kind: base.cache_fingerprint(metadata[kind], range(1, 151)) for kind in ('raw', 'sam')}
    expected_cache['observation'] = signatures['raw']
    require(document['cache'] == source_document['cache'] == expected_cache, 'Cache fingerprints mismatch')
    source_training = base.verify_training(source, source_document['config'], source_code, metadata, signatures)
    training = verify_training(out, source, document['config'], code, metadata, signatures)
    source_artifacts = {}
    for artifact in source_document['artifacts']:
        key = tuple(artifact[name] for name in ('seed', 'split', 'controller', 'budget_m'))
        require(key not in source_artifacts, f'Duplicate original trajectory metadata: {key}')
        source_artifacts[key] = artifact
    rows = {}
    for row in document['rows']:
        key = tuple(row[name] for name in ('seed', 'split', 'controller', 'budget_m', 'scene', 'readout'))
        require(key not in rows, f'Duplicate aggregate row: {key}')
        rows[key] = row
    expected = set(itertools.product(SEEDS, base.SPLITS, CONTROLLERS, BUDGETS))
    seen, seen_paths = set(), set()
    checked_rows = episodes = 0
    for artifact in document['artifacts']:
        key = tuple(artifact[name] for name in ('seed', 'split', 'controller', 'budget_m'))
        require(key in expected and key not in seen, f'Unexpected/duplicate rollout: {key}')
        seen.add(key)
        path = confined_path(out, artifact['path'], 'Rollout')
        require(path not in seen_paths, f'Duplicate rollout path: {path}')
        seen_paths.add(path)
        require(sha256(path) == artifact['sha256'], f'Rollout SHA256 mismatch: {path}')
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        audited = verify_rollout(arrays, base.SPLITS[artifact['split']], artifact['budget_m'], probabilities, path.name)
        verify_source_rollout(artifact, arrays, source, source_artifacts)
        checked_rows += verify_metrics(rows, artifact, audited)
        episodes += len(audited['labels'])
        if len(seen) % 25 == 0:
            print(f'Verified {len(seen)}/{len(expected)} ablation artifacts; {episodes:,} episodes', flush=True)
    require(seen == expected, 'Missing expected rollout artifacts')
    require(not rows, f'Unmatched aggregate rows: {len(rows)}')
    return {
        'passed': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'results_sha256': sha256(out / 'results.json'), 'selection_sha256': sha256(out / 'selection.json'),
        'source_results_sha256': sha256(source / 'results.json'),
        'source_selection_sha256': sha256(source / 'selection.json'),
        'verifier_sha256': sha256(Path(__file__)), 'source_sha256': code,
        'artifact_count': len(seen), 'episodes': episodes, 'aggregate_rows': checked_rows,
        'cache_files_verified': 300, 'training_manifests': training, 'source_training_manifests': source_training,
        'checks': ['source/protocol/artifact/cache/checkpoint SHA256', '90/10 development split and validation-selected cost',
                   'five seeds and all controller/budget/split/readout combinations', 'reproduced original checkpoint tensors',
                   'identical original controller paths and baseline readouts', '605 ordered starts per scene',
                   'independent lattice actions, boundaries, distance caps, horizon and stop reasons',
                   'Unknown and scene/class mapping', 'final/max-confidence/unique-view mean',
                   'all six evidence update sequences and operation counts', 'deterministic rule choices and tie handling',
                   'binary gates never fuse', 'per-scene and pooled accuracy/distance/rescue/harm/Unknown/stop aggregates'],
        'scope': 'Recorded neural actions and gate choices are checked against physical/evidence rules; rule decisions are independently derived. Neural predictions and validation rollouts are not independently re-inferred here; checkpoint decision replay is a separate audit.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'output/active_evidence')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/active_evidence_ablations')
    args = parser.parse_args()
    try:
        report = verify(args.out, args.source)
    except Exception as error:
        if args.out.is_dir():
            base.write_json(args.out / 'verification.json', {'passed': False, 'error_type': type(error).__name__, 'error': str(error)})
        raise
    base.write_json(args.out / 'verification.json', report)
    print({key: report[key] for key in ('passed', 'artifact_count', 'episodes', 'aggregate_rows')})


if __name__ == '__main__':
    main()
