"""Matched-data gate/embedding ablations, preserving the preceding experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .experiment import (ROOT, SEEDS, BUDGETS, COEFFICIENTS, CONFIG, Observations,
                         DEVELOPMENT, VALIDATION, TRAIN, SIMPLE, HARD, digest,
                         write, seed_all, fit, histories, rollout, training_identity)
from .model import classify, gate_update, state_input
from .ablation_model import FeatureGate, FeatureGain, RuleGate

READOUTS = ['final', 'max_confidence', 'mean', 'rule_latest', 'rule_agree',
            'gate2_full', 'gate3_full', 'gate2_noembed', 'gate3_noembed']
GATE_NAMES = READOUTS[3:]
CONTROLLERS = ['gain_full', 'joint_reference', 'gain_noembed', 'gain_noembed_fixed_cost']
MODEL_SPECS = {'gate3_full': (3, True), 'gate2_full': (2, True),
               'gate3_noembed': (3, False), 'gate2_noembed': (2, False)}


def code_identity():
    return {name: digest(Path(__file__).with_name(name)) for name in
            ('experiment.py', 'model.py', 'ablation_model.py', 'ablations.py', 'ABLATION_PROTOCOL.md')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def authenticate_source(source):
    verification = read(source / 'verification.json')
    for name in ('results', 'selection'):
        if verification.get(name + '_sha256') != digest(source / (name + '.json')):
            raise ValueError('Source verification identity mismatch: ' + name)
    if verification.get('passed') is not True:
        raise ValueError('Source verification must pass')
    selection = read(source / 'selection.json')
    if selection['config'] != CONFIG:
        raise ValueError('Source config mismatch')
    for name, expected in selection['code'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Source experiment code changed: ' + name)
    return selection, read(source / 'results.json')


def rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def weights(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def instantiate(name, device):
    if name.startswith('gate'):
        model = FeatureGate(*MODEL_SPECS[name])
    else:
        model = FeatureGain(embeddings=name == 'gain_full')
    return model.to(device)


def tensor_digest(*values):
    import hashlib
    h = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().numpy()
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def train(data, scenes, seed, path, source_path):
    reference = read(source_path / 'training.json')
    if reference['identity'] != training_identity(data, scenes, seed):
        raise ValueError('Reference training identity mismatch')
    for name in ('gate', 'gain_final'):
        if digest(source_path / (name + '.pt')) != reference['checkpoint_sha256'][name]:
            raise ValueError('Reference model hash mismatch')
    identity = {'config': CONFIG, 'code': code_identity(), 'training_scenes': scenes,
                'seed': seed, 'cache': data.fingerprints,
                'source_training_sha256': digest(source_path / 'training.json')}
    manifest_path = path / 'training.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest['identity'] != identity:
            raise ValueError('Ablation training changed; choose a new output directory')
        models = {}
        for name, expected in manifest['checkpoint_sha256'].items():
            checkpoint = path / (name + '.pt')
            if digest(checkpoint) != expected:
                raise ValueError('Ablation checkpoint hash mismatch')
            model = instantiate(name, data.device)
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=data.device))
            models[name] = model.eval()
        return models

    path.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    started = time.time()
    full_gate = instantiate('gate3_full', data.device)
    initial_gate = weights(full_gate)
    gx, gy = histories(data, scenes)
    gate_rng = rng_state()
    gate_loss = fit(full_gate, gx, gy)
    ids = data.ids(scenes)
    full_gain = instantiate('gain_full', data.device)
    initial_gain = weights(full_gain)
    sx = state_input(data.obs[ids], data.p[ids])
    sy, legal = data.correct[data.following(ids)], data.legal[ids % 605]
    gain_rng = rng_state()
    gain_loss = fit(full_gain, sx, sy, legal)
    models = {'gate3_full': full_gate, 'gain_full': full_gain}
    losses = {'gate3_full': gate_loss, 'gain_full': gain_loss}
    reproduced = {}
    for name, old, key in [('gate3_full', 'gate', 'gate_max_abs'),
                           ('gain_full', 'gain_final', 'gain_max_abs')]:
        expected = torch.load(source_path / (old + '.pt'), weights_only=True, map_location=data.device)
        difference = max(float((value - expected[k]).abs().max()) for k, value in models[name].state_dict().items())
        if difference > 1e-6:
            raise ValueError(f'Full control failed to reproduce {old}: {difference}')
        reproduced[key] = difference
    for name in ('gate2_full', 'gate3_noembed', 'gate2_noembed'):
        model = instantiate(name, data.device)
        model.load_state_dict(initial_gate)
        restore_rng(gate_rng)
        losses[name] = fit(model, gx, gy[:, :model.choices])
        models[name] = model
    model = instantiate('gain_noembed', data.device)
    model.load_state_dict(initial_gain)
    restore_rng(gain_rng)
    losses['gain_noembed'] = fit(model, sx, sy, legal)
    models['gain_noembed'] = model
    hashes = {}
    for name, model in models.items():
        torch.save(model.state_dict(), path / (name + '.pt'))
        hashes[name] = digest(path / (name + '.pt'))
    write(manifest_path, {'identity': identity, 'checkpoint_sha256': hashes, 'loss': losses,
                         'reproduced_reference': reproduced,
                         'input_pairing': {'gate_examples_sha256': tensor_digest(gx, gy),
                                           'gain_examples_sha256': tensor_digest(sx, sy, legal),
                                           'shared_initial_weights_and_minibatch_order': True},
                         'seconds': time.time() - started})
    print(f'Trained {len(scenes)} scenes / seed {seed}, reproduced controls {reproduced}, '
          f'{time.time() - started:.1f}s', flush=True)
    return models


def gain_rollout(data, scenes, models, budget, coefficient, seed):
    return rollout(data, scenes, {'gain_final': models['gain_noembed'], 'gate': models['gate3_full']},
                   'gain_final', budget, coefficient, seed)


@torch.no_grad()
def replay(data, path, models):
    """All readouts receive exactly the same acquired views, never future views."""
    history = torch.as_tensor(path['history'], device=data.device)
    moves = torch.as_tensor(path['moves'], device=data.device)
    starts = history[:, 0]
    n = len(starts)
    gates = {'rule_latest': RuleGate('latest'), 'rule_agree': RuleGate('agree'),
             **{name: models[name] for name in MODEL_SPECS}}
    memory_obs = [data.obs[starts].clone() for _ in GATE_NAMES]
    memory_p = [data.p[starts].clone() for _ in GATE_NAMES]
    choices = torch.full((n, len(GATE_NAMES), 5), -1, device=data.device, dtype=torch.long)
    operations = torch.zeros((n, len(GATE_NAMES), 3), device=data.device, dtype=torch.long)
    best = data.p[starts].clone()
    summed = best.clone()
    count = (best.sum(1) > 0).float()
    for t in range(5):
        rows = (moves > t).nonzero().flatten()
        if not len(rows):
            continue
        nxt = history[rows, t + 1]
        repeated = (history[rows, :t + 1] == nxt[:, None]).any(1)
        new_obs, new_p = data.obs[nxt], data.p[nxt]
        for j, name in enumerate(GATE_NAMES):
            mo, mp, op = gate_update(gates[name], memory_obs[j][rows], memory_p[j][rows],
                                      new_obs, new_p, repeated)
            memory_obs[j][rows], memory_p[j][rows] = mo, mp
            choices[rows, j, t] = op
            operations[rows, j, op] += 1
        novel = ~repeated & (new_p.sum(1) > 0)
        replace = novel & (new_p.max(1).values > best[rows].max(1).values)
        best[rows[replace]] = new_p[replace]
        summed[rows] += new_p * novel[:, None]
        count[rows] += novel
    final_ids = history[torch.arange(n, device=data.device), moves]
    probabilities = torch.stack((data.p[final_ids], best, summed / count.clamp_min(1)[:, None], *memory_p), 1)
    result = {name: path[name].copy() for name in
              ('starts', 'history', 'actions', 'distance_m', 'moves', 'stop_reason', 'labels')}
    result.update({name: value.cpu().numpy() for name, value in
                   {'predictions': classify(probabilities), 'final_probabilities': probabilities,
                    'gate_choices': choices, 'gate_operations': operations}.items()})
    return result


def metrics(result, scenes, controller, budget, seed, split):
    rows = []
    for scene in ['all', *scenes]:
        ix = slice(None) if scene == 'all' else slice(scenes.index(scene) * 605, (scenes.index(scene) + 1) * 605)
        truth, prediction = result['labels'][ix], result['predictions'][ix]
        base = prediction[:, 0] == truth
        for j, readout in enumerate(READOUTS):
            correct = prediction[:, j] == truth
            rows.append({'seed': seed, 'split': split, 'scene': scene, 'controller': controller,
                         'budget_m': budget, 'readout': readout, 'episodes': len(truth),
                         'accuracy': float(correct.mean()), 'distance_m': float(result['distance_m'][ix].mean()),
                         'moves': float(result['moves'][ix].mean()), 'unknown_rate': float((prediction[:, j] < 0).mean()),
                         'rescue_rate': float((correct & ~base).mean()), 'harm_rate': float((~correct & base).mean()),
                         'policy_stop_rate': float((result['stop_reason'][ix] == 1).mean()),
                         'gate_operation_counts': result['gate_operations'][ix, j - 3].sum(0).tolist() if j >= 3 else None})
    return rows


def development(out, source, device):
    original, _ = authenticate_source(source)
    data = Observations(TRAIN, device)
    rows = []
    for seed in SEEDS:
        models = train(data, DEVELOPMENT, seed, out / 'development' / f'seed_{seed}',
                       source / 'development' / f'seed_{seed}')
        for coefficient in COEFFICIENTS:
            for budget in BUDGETS[1:]:
                result = gain_rollout(data, VALIDATION, models, budget, coefficient, seed)
                accuracy = float((result['predictions'][:, 0] == result['labels']).mean())
                distance = float(result['distance_m'].mean())
                rows.append({'seed': seed, 'coefficient': coefficient, 'budget_m': budget,
                             'accuracy': accuracy, 'distance_m': distance, 'utility': accuracy - .001 * distance})
        write(out / 'development_progress.json', {'rows': rows})
        print(f'Ablation validation seed {seed} complete', flush=True)
    candidates = []
    for coefficient in COEFFICIENTS:
        selected = [r for r in rows if r['coefficient'] == coefficient]
        candidates.append({'coefficient': coefficient,
                           **{k: float(np.mean([r[k] for r in selected])) for k in ('utility', 'accuracy', 'distance_m')}})
    best = max(candidates, key=lambda r: (r['utility'], -r['distance_m'], r['coefficient']))
    selected = {'gain_full': original['selected']['gain_final'], 'gain_noembed': best['coefficient'],
                'gain_noembed_fixed_cost': original['selected']['gain_final']}
    write(out / 'selection.json', {'config': CONFIG, 'code': code_identity(), 'training_scenes': DEVELOPMENT,
                                  'validation_scenes': VALIDATION, 'selected': selected, 'candidates': candidates,
                                  'rows': rows, 'source_selection_sha256': digest(source / 'selection.json'),
                                  'test_data_loaded': False})
    print('Frozen ablation selection: ' + str(selected), flush=True)


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def final(out, source, device, audit=False):
    _, original = authenticate_source(source)
    selection = read(out / 'selection.json')
    if selection['config'] != CONFIG or selection['code'] != code_identity():
        raise ValueError('Selection code/config changed')
    if selection['source_selection_sha256'] != digest(source / 'selection.json'):
        raise ValueError('Source selection changed')
    # Test data is loaded only after internal selection is frozen.
    data = Observations(list(range(1, 151)), device)
    original_artifacts = {(a['seed'], a['split'], a['controller'], a['budget_m']): a for a in original['artifacts']}
    all_rows, artifacts = [], []
    if audit:
        document = read(out / 'results.json')
        if document.get('complete') is not True or document['code'] != code_identity():
            raise ValueError('Cannot replay incomplete/stale results')
        expected_artifacts = {a['path']: a for a in document['artifacts']}
    for seed in SEEDS:
        models = train(data, TRAIN, seed, out / 'final' / f'seed_{seed}', source / 'final' / f'seed_{seed}')
        for split, scenes in [('simple', SIMPLE), ('hard', HARD)]:
            for controller in CONTROLLERS:
                for budget in BUDGETS:
                    source_path = source_hash = None
                    if controller in ('gain_full', 'joint_reference'):
                        old_name = 'gain_final' if controller == 'gain_full' else 'gain_gate'
                        a = original_artifacts[seed, split, old_name, budget]
                        source_path, source_hash = a['path'], a['sha256']
                        if digest(source / source_path) != source_hash:
                            raise ValueError('Source rollout changed')
                        path_result = load_npz(source / source_path)
                    else:
                        path_result = gain_rollout(data, scenes, models, budget, selection['selected'][controller], seed)
                    result = replay(data, path_result, models)
                    # Full controls must preserve the historical gate output as well.
                    if not np.array_equal(result['predictions'][:, READOUTS.index('gate3_full')], path_result['predictions'][:, 3]):
                        raise ValueError('Full gate replay changed predictions')
                    relative = f'rollouts/{seed}_{split}_{controller}_{budget}.npz'
                    path = out / relative
                    if audit:
                        if digest(path) != expected_artifacts[relative]['sha256']:
                            raise ValueError('Changed artifact before model replay')
                        saved = load_npz(path)
                        if set(saved) != set(result):
                            raise ValueError('Model replay array keys differ')
                        for key in result:
                            if not np.array_equal(saved[key], result[key]):
                                raise ValueError(f'Model replay mismatch: {relative}/{key}')
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, **result)
                        all_rows.extend(metrics(result, scenes, controller, budget, seed, split))
                    artifacts.append({'path': relative, 'sha256': digest(path), 'seed': seed, 'split': split,
                                      'controller': controller, 'budget_m': budget,
                                      'source_path': source_path, 'source_sha256': source_hash})
            print(f'{"Replayed" if audit else "Evaluated"} ablations seed {seed} / {split}', flush=True)
        if not audit:
            write(out / 'results.json', {'config': CONFIG, 'code': code_identity(), 'readouts': READOUTS,
                                       'gate_names': GATE_NAMES, 'controllers': CONTROLLERS,
                                       'selection_sha256': digest(out / 'selection.json'), 'cache': data.fingerprints,
                                       'rows': all_rows, 'artifacts': artifacts, 'complete': seed == SEEDS[-1]})
    if audit:
        if set(expected_artifacts) != {a['path'] for a in artifacts}:
            raise ValueError('Model replay coverage mismatch')
        write(out / 'model_replay.json', {'passed': True, 'results_sha256': digest(out / 'results.json'),
                                        'selection_sha256': digest(out / 'selection.json'), 'code': code_identity(),
                                        'artifact_count': len(artifacts),
                                        'scope': 'All ablation gates rerun from checkpoints; new noembed action paths rerun. '
                                                 'Historical controller paths authenticated and reused, not reinferred.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'development', 'final', 'audit'], default='all')
    parser.add_argument('--source', type=Path, default=ROOT / 'output/active_evidence')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/active_evidence_ablations')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage in ('all', 'development'):
        development(args.out, args.source, args.device)
    if args.stage in ('all', 'final', 'audit'):
        final(args.out, args.source, args.device, audit=args.stage == 'audit')


if __name__ == '__main__':
    main()
