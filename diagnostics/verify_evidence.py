"""Recompute aggregate identities independently from the exported count tables."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(p):
    s = json.loads((p / 'summary.json').read_text())
    with (p / 'counts.csv').open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    scenes = [r for r in rows if r['scene'] != 'all']
    totals = [r for r in rows if r['scene'] == 'all' and r['subset'] == 'all']
    for t in totals:
        rs = [r for r in scenes if all(r[k] == t[k] for k in ('seed', 'split', 'mode', 'policy'))]
        require(len(rs) == 25, 'Missing scene rows')
        for k in t:
            if k not in ('seed', 'split', 'mode', 'policy', 'scene', 'subset'):
                require(math.isclose(sum(float(r[k]) for r in rs), float(t[k]), rel_tol=1e-10, abs_tol=1e-7), f'Scene sum mismatch {k}')
        for m in s['protocol']['methods']:
            require(int(t[m+'_correct'])-int(t['final_correct']) == int(t[m+'_rescue'])-int(t[m+'_harm']), 'Rescue/harm identity')
        require(int(t['oracle']) == int(t['final_correct']) + int(t['lost_correct']), 'Oracle identity')
    for e in s['summary']:
        rs = [r for r in rows if r['scene']=='all' and all(r[k]==e[k] for k in ('split','mode','policy','subset')) and int(r['seed']) in e['seeds']]
        for m in s['protocol']['methods']:
            acc = [int(r[m+'_correct']) / int(r['n']) for r in rs]
            delta = [100*(int(r[m+'_correct'])-int(r['final_correct']))/int(r['n']) for r in rs]
            for field, values in [('accuracy', acc), ('delta_vs_final_pp', delta)]:
                actual = e['methods'][m][field]
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if len(values)>1 else 0
                require(math.isclose(mean, actual['mean'], abs_tol=1e-12), 'Mean mismatch')
                require(math.isclose(sd, actual['sd'], abs_tol=1e-12), 'SD mismatch')
                if len(values)==5:
                    radius = 2.776445105*sd/math.sqrt(5)
                    require(all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip([mean-radius,mean+radius],actual['ci95_t'])), 'CI mismatch')
    probe = json.loads((p / 'matched_stop_probe.json').read_text())
    require(probe['base_summary_sha256'] == digest(p/'summary.json'), 'Probe source changed')
    for r in probe['rows']:
        for m in ('final','max_confidence','oracle'):
            require(r[m+'_correct']-r['native_correct'] == r[m+'_rescue']-r[m+'_harm'], 'Probe rescue/harm identity')
        if r['subset']=='all':
            a = next(x for x in totals if int(x['seed'])==r['seed'] and x['split']=='hard' and x['mode']=='native' and x['policy']==r['policy'])
            b = next(x for x in totals if int(x['seed'])==r['seed'] and x['split']=='hard' and x['mode']=='fixed_moves' and x['policy']==r['policy'])
            require(r['native_correct']==int(a['final_correct']) and r['final_correct']==int(b['final_correct']), 'Probe endpoint mismatch')
            require(r['oracle_correct']==int(b['oracle']), 'Probe oracle mismatch')
        if r['subset']=='immediate_stop':
            require(r['native_distance']==0 and r['oracle_harm']==0, 'Immediate-stop invariant')
    result = {'passed': True, 'scene_aggregate_groups': len(totals), 'summary_groups': len(s['summary']),
              'matched_probe_rows': len(probe['rows']), 'checked': ['scene sums', 'rescue/harm identities',
                'oracle identity', 'accuracy and paired delta means, SDs and t intervals', 'matched probe endpoints'],
              'input_sha256': {name: digest(p/name) for name in ('summary.json','counts.csv','matched_stop_probe.json')},
              'verifier_sha256': digest(Path(__file__))}
    (p/'aggregate_verification.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'output/evidence_fusion')
    verify(parser.parse_args().out)
