"""One or two seeds of short-budget GRU training with matched frozen controls.

This independent diagnostic does not resume the old 300k experiment queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'e2e_map/gru_quick.py', 'e2e_map/gru_quick_report.py', 'e2e_map/experiment.py', 'e2e_map/model.py',
    'e2e_map/images.py', 'e2e_map/evaluate.py', 'e2e_map/control.py',
    'ibe_map/model.py', 'ibe_map/experiment.py', 'ibe_map/verify_results.py',
    'map_baseline/model.py', 'map_baseline/environment.py',
    'map_baseline/features.py', 'map_baseline/checkpoints.py',
)
# The old trainers remain unchanged; both new arms explicitly disable TF32.
WORKER = (
    'import sys,runpy,torch; '
    'torch.backends.cuda.matmul.allow_tf32=False; '
    'torch.backends.cudnn.allow_tf32=False; '
    'sys.argv=sys.argv[1:]; '
    "runpy.run_module(sys.argv[0],run_name='__main__')"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def validate_request(variant, seeds, episodes):
    if variant not in ('memo_map', 'ibe_map'):
        raise ValueError('This diagnostic is limited to GRU policies')
    if not 1 <= len(seeds) <= 2 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('Specify one or two distinct nonnegative seeds')
    if not 1000 <= episodes <= 10000:
        raise ValueError('Short experiment budget must be between 1000 and 10000 episodes')


def execute(args):
    validate_request(args.variant, args.seeds, args.episodes)
    args.out = args.out.resolve()
    identity = {'variant': args.variant, 'seeds': args.seeds, 'episodes': args.episodes,
                'source_sha256': hashes()}
    if args.report_only:
        manifest = read_json(args.out / 'manifest.json')
        if any(manifest.get(k) != value for k, value in identity.items()):
            raise ValueError('Report request differs from the recorded experiment')
        from .gru_quick_report import build_report
        return build_report(args.out, args.variant, args.seeds, args.episodes)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new or empty output directory; old runs are never overwritten')
    args.out.mkdir(parents=True, exist_ok=True)
    from .pipeline import keep_awake, single_process

    def status(stage, **extra):
        value = {'stage': stage, 'variant': args.variant, 'seeds': args.seeds,
                 'episodes_each': args.episodes, 'updated_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
                 **extra}
        write_json(args.out / 'status.json', value)
        print(json.dumps(value), flush=True)

    def run(module, arguments, log, stage):
        if hashes() != identity['source_sha256']:
            raise ValueError('Experiment source changed during the short run')
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-c', WORKER, module, *map(str, arguments)]
        if module == 'ibe_map.verify_results':
            command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
        with log.open('w', encoding='utf8') as stream:
            stream.write('RUN ' + json.dumps(command) + '\n')
            stream.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status(stage, child_pid=child.pid, log=str(log.relative_to(ROOT)))
            if child.wait() != 0:
                raise RuntimeError(f'Job failed; inspect {log}')

    with single_process(args.out / 'pipeline.lock'), keep_awake():
        try:
            status('checking_caches')
            import torch
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('This short GPU diagnostic requires CUDA with BF16 support')
            from .control import _verify_control_cache
            control = ROOT / 'output/e2e_multiseed/control/cache'
            _verify_control_cache(control.parent)
            identity.update({
                'scope': 'short-budget diagnostic, not the paper 300000-episode experiment',
                'encoder_execution': 'E2E BF16, full activation checkpoint, microbatch32, fixed BN running stats',
                'tf32': False, 'evaluation_splits': ['simple', 'hard'],
                'epsilon_schedule': '1 to .05 over first 70% of the short budget in both arms',
                'control_cache_manifest_sha256': hashlib.sha256((control / 'control_cache_manifest.json').read_bytes()).hexdigest(),
            })
            write_json(args.out / 'manifest.json', identity)
            source = ROOT / ('output/ibe_map/cache' if args.variant == 'ibe_map' else 'output/map/cache')
            frozen_cache = control / ('sam' if args.variant == 'ibe_map' else 'raw')
            # Finish all requested training before looking at held-out outcomes.
            for seed in args.seeds:
                common = ['--variant', args.variant, '--seed', seed, '--episodes', args.episodes,
                          '--save-every', args.episodes, '--log-every', 1000, '--device', 'cuda']
                for arm in ('e2e', 'frozen'):
                    base = args.out / f'seed_{seed}' / arm
                    if arm == 'e2e':
                        module = 'e2e_map.experiment'
                        command = [*common, '--cache', source, '--out', base / 'train', '--microbatch', 32]
                    else:
                        module = 'ibe_map.experiment'
                        command = ['train', *common, '--phase', 'full', '--cache', frozen_cache, '--out', base / 'train']
                    run(module, command, base / 'training.log', f'training_{arm}_seed_{seed}')
            for seed in args.seeds:
                for arm in ('e2e', 'frozen'):
                    base = args.out / f'seed_{seed}' / arm
                    evaluation = base / 'evaluation'
                    if arm == 'e2e':
                        module = 'e2e_map.evaluate'
                        command = ['--checkpoint', base / 'train/last.pt', '--out', evaluation,
                                   '--splits', 'simple', 'hard', '--device', 'cuda']
                        raw = ROOT / 'output/map/cache'
                    else:
                        module = 'ibe_map.experiment'
                        command = ['evaluate', '--policies', args.variant, '--out', evaluation,
                                   '--raw-cache', control / 'raw', '--sam-cache', control / 'sam',
                                   '--' + args.variant.replace('_', '-') + '-checkpoint', base / 'train/last.pt',
                                   '--splits', 'simple', 'hard', '--seed', seed, '--no-report', '--device', 'cuda']
                        raw = control / 'raw'
                    run(module, command, base / 'evaluation.log', f'evaluating_{arm}_seed_{seed}')
                    run('ibe_map.verify_results', ['--out', evaluation, '--raw-cache', raw],
                        base / 'verification.log', f'verifying_{arm}_seed_{seed}')
            from .gru_quick_report import build_report
            result = build_report(args.out, args.variant, args.seeds, args.episodes)
            status('complete', active_training_jobs=0, summary=str(args.out / 'summary.json'))
            return result
        except BaseException as error:
            status('failed', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('memo_map', 'ibe_map'), default='ibe_map')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/gru_quick_20261024')
    parser.add_argument('--report-only', action='store_true')
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
