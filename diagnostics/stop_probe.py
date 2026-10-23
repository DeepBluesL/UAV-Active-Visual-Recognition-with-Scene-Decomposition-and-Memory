"""Pair native and forced-three-move records at identical starts.

This is a retrospective intervention diagnostic, not an equal-cost comparison
and not an evaluation of a deployable adaptive stopping rule.
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from diagnostics.evidence_fusion import ROOT, CLASSES, label_for_scene, predict_readouts, sha256, stats


def run(out):
    base = json.loads((out / 'summary.json').read_text(encoding='utf8'))
    rows = []
    for seed in base['seeds']:
        path = Path(base['source']) / f'seed_{seed}/evaluation/trajectories.jsonl.gz'
        if sha256(path) != base['source_sha256'][str(path.resolve())]:
            raise ValueError('Diagnostic source changed')
        native, totals = {}, defaultdict(Counter)
        with gzip.open(path, 'rt', encoding='utf8') as f:
            for line in f:
                if '"split":"hard"' not in line:
                    continue
                t = json.loads(line)
                if t['policy'] == 'stationary':
                    continue
                key = (t['policy'], t['scene'], t['start_x'], t['start_y'], t['start_z'])
                if t['mode'] == 'native':
                    if key in native:
                        raise ValueError('Duplicate native key')
                    native[key] = (t['final_correct'], t['distance_m'], t['actions'] == [24])
                    continue
                if t['mode'] != 'fixed_moves' or len(t['actions']) != 4 or t['actions'][-1] != 24:
                    raise ValueError('Not a fixed-three-move record')
                old_correct, old_distance, immediate = native.pop(key)
                pred, info = predict_readouts(t['positions'], t['probabilities'], t['actions'])
                truth = label_for_scene(t['scene'])
                correct = {'final': pred['final'] == truth, 'max_confidence': pred['max_confidence'] == truth,
                           'oracle': CLASSES[truth] in t['predictions']}
                c = Counter(n=1, native_correct=int(old_correct), native_distance=old_distance,
                            fixed_distance=t['distance_m'])
                for m, good in correct.items():
                    c[m + '_correct'] = int(good)
                    c[m + '_rescue'] = int(good and not old_correct)
                    c[m + '_harm'] = int(old_correct and not good)
                subsets = ['all', 'immediate_stop' if immediate else 'moved_before_stopping']
                for subset in subsets:
                    totals[(t['policy'], subset)].update(c)
        if native:
            raise ValueError('Unpaired native records')
        for (policy, subset), c in totals.items():
            if subset == 'all' and c['n'] != 15125:
                raise ValueError('Incomplete matched starts')
            rows.append({'seed': seed, 'policy': policy, 'subset': subset, **c})
        print(f'Matched stop probe seed {seed}', flush=True)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['policy'], r['subset'])].append(r)
    summary = []
    for (policy, subset), rs in sorted(groups.items()):
        if policy == 'sweep':
            numeric = set(rs[0]) - {'seed', 'policy', 'subset'}
            if any(any(r[k] != rs[0][k] for k in numeric) for r in rs[1:]):
                raise ValueError('Deterministic sweep changed')
            rs = rs[:1]
        summary.append({'policy': policy, 'subset': subset, 'seeds': [r['seed'] for r in rs],
                        'n': [r['n'] for r in rs],
                        'metrics': {k: stats([r[k] / r['n'] for r in rs]) for k in rs[0]
                                    if k not in ('seed', 'policy', 'subset', 'n')},
                        'delta_pp': {m: stats([100*(r[m+'_correct']-r['native_correct'])/r['n'] for r in rs])
                                     for m in ('final', 'max_confidence', 'oracle')}})
    result = {'protocol': 'Same policy/seed/scene/start; hard only. All fixed records acquire three movements. '
                          'Immediate-stop subset is defined by native actions==[STOP], without labels. '
                          'Extra distance differs; this is not an equal-budget policy comparison. '
                          'Native moved subset can differ even before final stopping due to forced STOP after three moves.',
              'rows': rows, 'summary': summary, 'passed': True,
              'base_summary_sha256': sha256(out / 'summary.json'), 'script_sha256': sha256(Path(__file__))}
    (out / 'matched_stop_probe.json').write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/evidence_fusion')
    run(parser.parse_args().out)
