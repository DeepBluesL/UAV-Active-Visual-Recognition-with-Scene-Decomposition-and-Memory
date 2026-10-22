"""Run, verify, summarize, and optionally publish the matched E2E experiment.

Publishing is opt-in. No result is published until all ten complete training
runs and independent trajectory checks pass. Existing frozen outputs are read-only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ibe_map.multiseed import cache_identity, read_json, sha256, write_json
from map_baseline.environment import TRAIN, SIMPLE, HARD, Protocol

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ('map', 'ibe_map')
REPOSITORY = 'https://github.com/DeepBluesL/UAV-Active-Visual-Recognition-with-Scene-Decomposition-and-Memory.git'
TRAIN_SOURCES = ('e2e_map/experiment.py', 'e2e_map/model.py', 'e2e_map/images.py',
                 'e2e_map/control.py',
                 'map_baseline/environment.py', 'map_baseline/model.py', 'map_baseline/experiment.py',
                 'map_baseline/features.py', 'map_baseline/checkpoints.py',
                 'ibe_map/model.py', 'ibe_map/experiment.py')
PUBLISH_SOURCES = ('e2e_map/__init__.py', 'e2e_map/model.py', 'e2e_map/images.py',
    'e2e_map/experiment.py', 'e2e_map/evaluate.py', 'e2e_map/pipeline.py',
    'e2e_map/control.py',
    'e2e_map/report.py', 'e2e_map/publish_results.py', 'e2e_map/test_model.py',
    'e2e_map/test_experiment.py', 'e2e_map/test_report.py', 'e2e_map/test_publish_results.py',
    'e2e_map/test_pipeline.py', 'e2e_map/test_control.py', 'e2e_map/README.md',
    'run_e2e.ps1', 'README.md', 'README.zh-CN.md')
PUBLISH_RESULTS = ('docs/e2e_results/README.md', 'docs/e2e_results/metrics.json',
                   'docs/e2e_results/accuracy_return.png')


def publication_identity(args):
    identity = {'source_sha256': {p: sha256(ROOT / p) for p in PUBLISH_SOURCES},
                'verifier_sha256': sha256(ROOT / 'ibe_map/verify_results.py'),
                'base_commit': git('rev-parse', 'HEAD')}
    path = args.out / 'publication_manifest.json'
    if path.exists():
        # Publishing creates one new local commit; a retry may start at that commit.
        expected = read_json(path)
        if any(expected[k] != identity[k] for k in ('source_sha256', 'verifier_sha256')):
            raise ValueError('Evaluation/publication source changed since experiment launch')
    else:
        write_json(path, identity)
    return read_json(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf8',
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'}).strip()


def manifest(args):
    baseline = read_json(args.frozen / 'summary.json')
    if baseline['seeds'] != list(SEEDS) or baseline['protocol']['requested_training_episodes'] != 300000:
        raise ValueError('Expected the existing five-seed 300k frozen control')
    if not read_json(args.frozen / 'summary_verification.json').get('passed'):
        raise ValueError('Frozen summary has no passing independent verification')
    caches = {kind: cache_identity(ROOT / path) for kind, path in (
        ('raw', 'output/map/cache'), ('sam', 'output/ibe_map/cache'))}
    identity = {'schema_version': 1, 'seeds': list(SEEDS), 'variants': list(VARIANTS),
        'episodes': 300000, 'save_every': 30000, 'microbatch': args.microbatch,
        'encoder_precision': 'bfloat16; FP32 features; matching BF16 frozen controls trained separately',
        'device': args.device, 'python': str(Path(sys.executable).resolve()),
        'source_sha256': {name: sha256(ROOT / name) for name in TRAIN_SOURCES},
        'frozen_summary_sha256': sha256(args.frozen / 'summary.json'),
        'frozen_verification_sha256': sha256(args.frozen / 'summary_verification.json'),
        'caches': caches, 'training_scenes': TRAIN, 'simple_scenes': SIMPLE, 'hard_scenes': HARD,
        'protocol': Protocol().as_dict(),
        'selection': 'final300k; all ten E2E and ten BF16 frozen training jobs finish before any hard evaluation',
        'batchnorm': 'ImageNet running statistics fixed, all affine parameters trainable',
        'reference': 'matched BF16 frozen MAP/IBE primary; existing FP16 frozen five-seed results are historical context'}
    p = args.out / 'run_manifest.json'
    if p.exists():
        if read_json(p) != identity:
            raise ValueError('Experiment identity changed; use a new output directory')
    else:
        write_json(p, identity)
    return identity


def status(args, stage, **fields):
    value = {'stage': stage, 'updated_local': time.strftime('%Y-%m-%d %H:%M:%S'),
             'pid': os.getpid(), **fields}
    write_json(args.out / 'status.json', value)
    print(json.dumps(value), flush=True)


def run(args, command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf8') as log:
        log.write('\nRUN ' + json.dumps([str(v) for v in command]) + '\n'); log.flush()
        subprocess.run([str(v) for v in command], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1'})


def complete_train(base, seed, variant):
    p = base / 'train/training.json'
    if not p.exists():
        return False
    training = read_json(p); c = training['config']
    if c['seed'] != seed or c['variant'] != variant or c['requested_episodes'] != 300000:
        raise ValueError('Wrong training identity')
    if not training.get('complete'):
        return False
    if not 300000 <= training['completed_episodes'] <= 300063:
        raise ValueError('Wrong final training budget')
    if not all(value > 0 for value in training['backbone_parameter_l2_change'].values()):
        raise ValueError('Some backbone stages did not change')
    if not training['batchnorm_running_stats_unchanged']:
        raise ValueError('BatchNorm running statistics changed')
    if c['training_scenes'] != TRAIN or c['validation_scenes']:
        raise ValueError('Wrong training/validation scenes')
    if not (base / 'train/last.pt').exists():
        raise ValueError('Missing complete checkpoint')
    return True


def verify_current(base):
    path = base / 'evaluation/verification.json'
    if not path.exists():
        return False
    v = read_json(path)
    if not v.get('passed') or any(v['counts'].get(k) != 105875
                                  for k in ('trajectories', 'csv_rows', 'unique_episode_keys')):
        return False
    result = read_json(base / 'evaluation/results.json')
    checkpoint = base / 'train/last.pt'
    c = read_json(base / 'train/training.json')['config']
    if result['protocol']['checkpoint_sha256'] != {c['variant']: sha256(checkpoint)}:
        return False
    for name in ('results.json', 'evaluation.csv', 'trajectories.jsonl.gz'):
        p = (base / 'evaluation' / name).resolve()
        if v['input_sha256'].get(str(p)) != sha256(p):
            return False
    return True


def publication_base(args, head, remote):
    """Allow retrying this pipeline's own committed-but-unpushed result."""
    if remote == head:
        return head
    pending = args.out / 'publication_pending.json'
    if pending.is_file():
        attempt = read_json(pending)
        if (attempt.get('commit') == head and attempt.get('base_commit') == remote
                and attempt.get('repository') == REPOSITORY
                and attempt.get('files') == list(PUBLISH_SOURCES + PUBLISH_RESULTS)
                and git('rev-parse', 'HEAD^') == remote):
            return remote
    raise ValueError('Remote main differs from local HEAD; refusing automatic overwrite')


def publish(args):
    """Explicit, bounded commit of source and compact verified results only."""
    from .report import build_summary
    publication_identity(args)
    run(args, [sys.executable, '-m', 'e2e_map.control', '--stage', 'summarize',
        '--out', args.out / 'control', '--device', args.device], args.out / 'control_summary.log')
    summary = build_summary(args.out / 'control/summary.json', args.out,
                            historical_summary_path=args.frozen / 'summary.json')
    if any(not verify_current(args.out / f'seed_{s}' / p) for s in SEEDS for p in VARIANTS):
        raise ValueError('Publishing requires current independent trajectory verification')
    from .publish_results import export_results
    export_results(summary, args.out, ROOT / 'docs/e2e_results')
    shutil.copyfile(ROOT / 'docs/e2e_results/accuracy_return.png', args.out / 'accuracy_return.png')
    if git('remote', 'get-url', 'origin') != REPOSITORY or git('branch', '--show-current') != 'main':
        raise ValueError('Repository or branch changed; refusing automatic publication')
    # Never commit unrelated staged work. Worktree changes outside these paths
    # are left untouched; no reset/clean/force operation is used.
    if git('diff', '--cached', '--name-only'):
        raise ValueError('Index already has staged changes; refusing to include other work')
    head = git('rev-parse', 'HEAD')
    remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
    base_commit = publication_base(args, head, remote)
    files = list(PUBLISH_SOURCES + PUBLISH_RESULTS)
    if any((ROOT / name).stat().st_size > 10_000_000 for name in files):
        raise ValueError('Unexpected large publication file')
    subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
    staged = git('diff', '--cached', '--name-only').splitlines()
    if set(staged) - set(files):
        raise ValueError('Unexpected staged path')
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    if staged:
        if base_commit != head:
            raise ValueError('Published files changed after the pending commit; refusing a second automatic commit')
        subprocess.run(['git', 'commit', '-m', 'Train ResNet50 end to end and compare matched five-seed MAP/IBE results'], cwd=ROOT, check=True)
    write_json(args.out / 'publication_pending.json', {
        'base_commit': base_commit, 'commit': git('rev-parse', 'HEAD'),
        'repository': REPOSITORY, 'files': files})
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=ROOT, check=True,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'})
    commit = git('rev-parse', 'HEAD')
    if git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit:
        raise ValueError('Published remote head verification failed')
    write_json(args.out / 'publication.json', {'commit': commit, 'repository': REPOSITORY,
        'url': REPOSITORY.removesuffix('.git') + '/commit/' + commit,
        'files': files, 'verified': True})


@contextlib.contextmanager
def single_process(path):
    # OS lock is released automatically even if a long run is interrupted.
    handle = path.open('a+b'); handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        handle.close()


@contextlib.contextmanager
def keep_awake():
    """Prevent automatic system sleep only while this long computation runs.

    The screen may turn off. Explicit user sleep/power requests are not blocked.
    Windows also releases this per-thread request if the process terminates.
    """
    if os.name != 'nt':
        yield
        return
    import ctypes
    kernel = ctypes.windll.kernel32
    active = kernel.SetThreadExecutionState(0x80000001)  # CONTINUOUS | SYSTEM_REQUIRED
    try:
        yield
    finally:
        if active:
            kernel.SetThreadExecutionState(0x80000000)


def execute(args):
    args.out = args.out.resolve()
    args.frozen = args.frozen.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with single_process(args.out / 'pipeline.lock'), keep_awake():
        status(args, 'validating')
        identity = manifest(args)
        if args.stage == 'plan':
            status(args, 'planned', e2e_training_jobs=10, matched_frozen_training_jobs=10)
            return
        try:
            publication_identity(args)
            if args.stage in ('all', 'train'):
                status(args, 'preparing_matched_controls')
                run(args, [sys.executable, '-m', 'e2e_map.control', '--stage', 'prepare',
                    '--out', args.out / 'control', '--device', args.device], args.out / 'control_prepare.log')
                status(args, 'training_matched_controls', training_jobs=10)
                run(args, [sys.executable, '-m', 'e2e_map.control', '--stage', 'train',
                    '--out', args.out / 'control', '--device', args.device], args.out / 'control_train.log')
                for job, (seed, variant) in enumerate((s, p) for s in SEEDS for p in VARIANTS):
                    base = args.out / f'seed_{seed}' / variant
                    if complete_train(base, seed, variant):
                        continue
                    status(args, 'training', job=job + 1, total_jobs=10, seed=seed, policy=variant,
                           training_log=str(base / 'training.log'))
                    run(args, [sys.executable, '-m', 'e2e_map.experiment', '--variant', variant,
                        '--seed', seed, '--episodes', '300000', '--out', base / 'train',
                        '--microbatch', args.microbatch, '--device', args.device, '--resume'], base / 'training.log')
                    if not complete_train(base, seed, variant):
                        raise ValueError('Training subprocess did not complete its full budget')
                status(args, 'trained', training_jobs=10)
            if args.stage in ('all', 'evaluate'):
                if not all(complete_train(args.out / f'seed_{s}' / p, s, p) for s in SEEDS for p in VARIANTS):
                    raise ValueError('All ten trainings must finish before held-out evaluation')
                status(args, 'evaluating_matched_controls')
                run(args, [sys.executable, '-m', 'e2e_map.control', '--stage', 'evaluate',
                    '--out', args.out / 'control', '--device', args.device], args.out / 'control_evaluate.log')
                run(args, [sys.executable, '-m', 'e2e_map.control', '--stage', 'summarize',
                    '--out', args.out / 'control', '--device', args.device], args.out / 'control_summary.log')
                for seed in SEEDS:
                    for variant in VARIANTS:
                        base = args.out / f'seed_{seed}' / variant
                        if verify_current(base):
                            continue
                        status(args, 'evaluating', seed=seed, policy=variant)
                        run(args, [sys.executable, '-m', 'e2e_map.evaluate', '--checkpoint', base / 'train/last.pt',
                            '--out', base / 'evaluation', '--device', args.device], base / 'evaluation.log')
                        run(args, [sys.executable, '-m', 'ibe_map.verify_results', '--out', base / 'evaluation'],
                            base / 'verification.log')
                        if not verify_current(base):
                            raise ValueError('Independent trajectory verification did not match final checkpoint')
            if args.stage in ('all', 'report'):
                if not all(verify_current(args.out / f'seed_{s}' / p) for s in SEEDS for p in VARIANTS):
                    raise ValueError('Report requires complete independently verified evaluations')
                # Immutable training source/control identities must still match.
                manifest(args)
                run(args, [sys.executable, '-m', 'e2e_map.control', '--stage', 'summarize',
                    '--out', args.out / 'control', '--device', args.device], args.out / 'control_summary.log')
                status(args, 'summarizing')
                run(args, [sys.executable, '-m', 'e2e_map.report', '--out', args.out,
                           '--frozen', args.out / 'control/summary.json',
                           '--historical', args.frozen / 'summary.json'], args.out / 'report.log')
                from .publish_results import export_results
                export_results(read_json(args.out / 'comparison_summary.json'), args.out, ROOT / 'docs/e2e_results')
                shutil.copyfile(ROOT / 'docs/e2e_results/accuracy_return.png', args.out / 'accuracy_return.png')
            if args.publish and args.stage in ('all', 'report', 'publish'):
                manifest(args)
                status(args, 'publishing')
                publish(args)
                status(args, 'complete', published=True, publication=read_json(args.out / 'publication.json'))
            elif args.stage in ('all', 'report'):
                status(args, 'complete', published=False)
        except BaseException as exc:
            status(args, 'failed', error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/e2e_multiseed')
    parser.add_argument('--frozen', type=Path, default=ROOT / 'output/ibe_multiseed')
    parser.add_argument('--stage', choices=('all', 'plan', 'train', 'evaluate', 'report', 'publish'), default='all')
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--publish', action='store_true', help='After all checks, commit bounded source/results and push existing origin main')
    execute(parser.parse_args())
