"""Single-seed MAP/IBE reference experiment; never report seed uncertainty."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import pipeline as p
from . import control as c
from .report import _load_e2e, _validate_match

SEED = 42
POLICIES = ('map', 'ibe_map')
SOURCES = p.PUBLISH_SOURCES + ('e2e_map/pilot.py', 'e2e_map/test_pilot.py', 'e2e_map/PILOT.md')
RESULTS = ('docs/e2e_pilot/README.md', 'docs/e2e_pilot/metrics.json')


def identity(args):
    # Preserve the original five-seed manifest as provenance; explicitly record
    # the user-requested reduction separately instead of rewriting its history.
    p.manifest(args)
    value = {'mode': 'single_seed_reference', 'seeds': [SEED], 'policies': list(POLICIES),
        'episodes_per_run': 300000, 'cancelled_e2e_seeds': [43, 44, 45, 46],
        'source_sha256': {name: p.sha256(p.ROOT / name) for name in SOURCES},
        'verifier_sha256': p.sha256(p.ROOT / 'ibe_map/verify_results.py'),
        'original_manifest_sha256': p.sha256(args.out / 'run_manifest.json'),
        'note': 'Reuse current MAP and completed frozen controls; only seed 42 is evaluated and compared.'}
    path = args.out / 'pilot_manifest.json'
    if path.exists() and p.read_json(path) != value:
        raise ValueError('Pilot identity changed; review before continuing')
    p.write_json(path, value)
    return value


def status(args, stage, **values):
    p.status(args, stage, mode='single_seed_reference', seeds=[SEED],
             e2e_training_jobs=2, cancelled_e2e_jobs=8, **values)


def wait_existing(args):
    """Wait on the existing Windows child without stopping or duplicating it."""
    if not args.follow_pid:
        return
    if os.name != 'nt':
        raise ValueError('--follow-pid is supported only on Windows')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, args.follow_pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # Process already exited; inspect its checkpoint below.
            return
        raise OSError(error, 'Cannot attach to existing MAP training')
    try:
        status(args, 'training', job=1, policy='map', existing_training_pid=args.follow_pid)
        while True:
            result = kernel.WaitForSingleObject(handle, 10000)
            if result == 0:
                return
            if result != 258:
                raise OSError('Unexpected wait status for existing training process')
    finally:
        kernel.CloseHandle(handle)


def evaluate_control(args):
    root = args.out / 'control'
    c._verify_control_cache(root)
    base = root / f'seed_{SEED}'
    for policy in POLICIES:
        cache = root / 'cache' / ('raw' if policy == 'map' else 'sam')
        if not c._complete_job(base / policy, SEED, policy, cache, args.device):
            raise ValueError('Selected frozen control is incomplete')
    if c._evaluation_current(base, SEED):
        return
    out = base / 'evaluation'
    p.run(args, [sys.executable, '-m', 'ibe_map.experiment', 'evaluate',
        '--policies', *POLICIES, '--raw-cache', root / 'cache/raw', '--sam-cache', root / 'cache/sam',
        '--map-checkpoint', base / 'map/train/last.pt',
        '--ibe-map-checkpoint', base / 'ibe_map/train/last.pt', '--out', out,
        '--seed', SEED, '--device', args.device, '--no-report'], base / 'evaluation.log')
    p.run(args, [sys.executable, '-m', 'ibe_map.verify_results', '--out', out,
        '--raw-cache', root / 'cache/raw', '--data', p.ROOT / 'dataset'], base / 'verification.log')
    p.write_json(out / 'postprocess_identity.json', {
        'verify_results_sha256': p.sha256(p.ROOT / 'ibe_map/verify_results.py'),
        'verification_sha256': p.sha256(out / 'verification.json'),
        'note': 'Independent trajectory verifier; single-seed reference.'})
    if not c._evaluation_current(base, SEED):
        raise ValueError('Selected frozen control evaluation is not independently verified')


def differences(groups):
    keys = ('accuracy', 'return_mean', 'distance_m', 'generalization_gap_return')
    delta = {policy: {key: groups[f'e2e_{policy}']['hard'][key]
        - groups[f'frozen_{policy}']['hard'][key] for key in keys} for policy in POLICIES}
    return {'e2e_minus_frozen': delta,
            'difference_in_differences': {key: delta['ibe_map'][key] - delta['map'][key] for key in keys}}


def summarize(args):
    identity(args)
    root = args.out / 'control'
    c._verify_control_cache(root)
    base = root / f'seed_{SEED}'
    if not c._evaluation_current(base, SEED):
        raise ValueError('Control evaluation is missing or stale')
    groups, records = {}, []
    proof_paths = {args.out / 'pilot_manifest.json', base / 'evaluation/verification.json',
                   base / 'evaluation/results.json', root / 'cache/control_cache_manifest.json'}
    for policy in POLICIES:
        cache = root / 'cache' / ('raw' if policy == 'map' else 'sam')
        frozen, result, _ = c._verify_job_and_evaluation(root, SEED, policy, cache)
        ebase = args.out / f'seed_{SEED}' / policy
        if not p.complete_train(ebase, SEED, policy) or not p.verify_current(ebase):
            raise ValueError('End-to-end training or evaluation is incomplete')
        e2e = _load_e2e(args.out, SEED, policy, 300000, 15125)
        _validate_match(frozen['config'], e2e['training']['config'], e2e['results'], SEED, policy)
        for regime, rows in (('frozen', result['metrics']), ('e2e', e2e['results']['metrics'])):
            groups[f'{regime}_{policy}'] = {row['split']: dict(row) for row in rows if row['policy'] == policy}
            block = groups[f'{regime}_{policy}']
            block['hard']['generalization_gap_return'] = block['train']['return_mean'] - block['hard']['return_mean']
        for regime, job, training in (('frozen', base / policy, frozen), ('e2e', ebase, e2e['training'])):
            records.append({'regime': regime, 'policy': policy, 'seed': SEED,
                'completed_episodes': training['completed_episodes'], 'updates': training['updates'],
                'elapsed_seconds': training['elapsed_seconds'],
                'checkpoint_sha256': p.sha256(job / 'train/last.pt'),
                'training_sha256': p.sha256(job / 'train/training.json')})
            proof_paths.update((job / 'train/last.pt', job / 'train/training.json'))
        proof_paths.update(ebase / 'evaluation' / name for name in
                           ('results.json', 'evaluation.csv', 'trajectories.jsonl.gz', 'verification.json'))
    proof_paths.update(base / 'evaluation' / name for name in
                       ('evaluation.csv', 'trajectories.jsonl.gz', 'postprocess_identity.json'))
    result = {'status': 'complete_single_seed_reference', 'seeds': [SEED], 'training_budget': 300000,
        'verified_trajectories': 423500, 'groups': groups, 'training_runs': records,
        **differences(groups), 'uncertainty': None,
        'limitations': ['Single seed only: no across-seed SD, confidence interval, or significance claim.',
            'Matched BF16 frozen and trainable ResNet50; identical scenes, rewards and starting viewpoints.',
            'Official pretrained SAM is not fine-tuned on UEVAVD.',
            'The IBE versus MAP comparison combines SAM and GRU; their individual contributions are not isolated.']}
    p.write_json(args.out / 'pilot_summary.json', result)
    proof_paths.add(args.out / 'pilot_summary.json')
    p.write_json(args.out / 'pilot_verification.json', {'passed': True,
        'trajectories': 423500, 'input_sha256': {str(path): p.sha256(path) for path in sorted(proof_paths)}})
    destination = p.ROOT / 'docs/e2e_pilot'
    destination.mkdir(parents=True, exist_ok=True)
    p.write_json(destination / 'metrics.json', result)
    rows = []
    for key, label in (('frozen_map', 'BF16 冻结 MAP'), ('e2e_map', '端到端 MAP'),
                       ('frozen_ibe_map', 'BF16 冻结 SAM+GRU'), ('e2e_ibe_map', '端到端 SAM+GRU')):
        h = groups[key]['hard']
        rows.append(f"| {label} | {100*h['accuracy']:.2f} | {h['return_mean']:.4f} | {h['distance_m']:.2f} |")
    changes = []
    for policy in POLICIES:
        d = result['e2e_minus_frozen'][policy]
        changes.append(f"- {policy}：准确率变化 {100*d['accuracy']:+.2f} 个百分点；回报变化 {d['return_mean']:+.4f}；移动距离变化 {d['distance_m']:+.2f} m。")
    did = result['difference_in_differences']
    document = '\n'.join(['# ResNet50 端到端：单种子初步对照', '',
        '按用户要求，从五种子计划缩减为 seed 42 的 MAP 与 SAM+GRU 两次端到端训练。每次 30 万回合；与相同 seed、相同 BF16 精度的冻结版本比较。其余八次端到端训练已取消，已完成的冻结训练保留。', '',
        '**这是单种子参考，不是五种子统计结论；不报告跨种子标准差、置信区间或显著性。**', '',
        '| 方法 | Hard 准确率（%） | 折扣回报 | 移动距离（m） |', '|---|---:|---:|---:|', *rows, '',
        '## 初步差值（端到端 − 冻结）', '', *changes, '',
        f"SAM+GRU 相对 MAP 的优势变化：准确率 {100*did['accuracy']:+.2f} 个百分点，回报 {did['return_mean']:+.4f}。这里只描述这一个 seed 的差值，不能据此认定稳定改善。距离增加意味着飞得更远，应与准确率和回报一起解读。", '',
        '四组策略均使用完整 train/simple/hard 起点及 hard 固定三步诊断，共 423,500 条轨迹，已通过独立核验。SAM 仍为官方预训练版本，未做 UEVAVD 微调。', '',
        '[完整指标与模型哈希](metrics.json) · [运行说明](../../e2e_map/PILOT.md)', ''])
    (destination / 'README.md').write_text(document, encoding='utf8')
    return result


def publish(args):
    identity(args)
    summarize(args)
    files = list(SOURCES + RESULTS)
    if p.git('remote', 'get-url', 'origin') != p.REPOSITORY or p.git('branch', '--show-current') != 'main':
        raise ValueError('Publication repository or branch changed')
    if p.git('diff', '--cached', '--name-only'):
        raise ValueError('Unrelated staged changes; refusing automatic publication')
    head, remote = p.git('rev-parse', 'HEAD'), p.git('ls-remote', 'origin', 'refs/heads/main').split()[0]
    pending_path = args.out / 'pilot_publication_pending.json'
    if head != remote:
        pending = p.read_json(pending_path) if pending_path.is_file() else {}
        if (pending != {'commit': head, 'base': remote, 'files': files}
                or p.git('rev-parse', 'HEAD^') != remote):
            raise ValueError('Remote/local history differs; refusing automatic overwrite')
    if any((p.ROOT / name).stat().st_size > 10000000 for name in files):
        raise ValueError('Unexpected large publication file')
    subprocess.run(['git', 'add', '--', *files], cwd=p.ROOT, check=True)
    staged = p.git('diff', '--cached', '--name-only').splitlines()
    if set(staged) - set(files):
        raise ValueError('Unexpected staged files')
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=p.ROOT, check=True)
    if staged:
        if head != remote:
            raise ValueError('Files changed after pending publication commit')
        subprocess.run(['git', 'commit', '-m', 'Add single-seed end-to-end ResNet50 reference comparison'], cwd=p.ROOT, check=True)
    commit = p.git('rev-parse', 'HEAD')
    p.write_json(pending_path, {'commit': commit, 'base': remote, 'files': files})
    p.git('push', 'origin', 'HEAD:main')
    if p.git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit:
        raise ValueError('Remote commit verification failed')
    p.write_json(args.out / 'publication.json', {'verified': True, 'mode': 'single_seed_reference',
        'commit': commit, 'url': p.REPOSITORY.removesuffix('.git') + '/commit/' + commit})


def execute(args):
    args.out, args.frozen = args.out.resolve(), args.frozen.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with p.single_process(args.out / 'pipeline.lock'), p.keep_awake():
        try:
            identity(args)
            wait_existing(args)
            for number, policy in enumerate(POLICIES, 1):
                base = args.out / f'seed_{SEED}' / policy
                if not p.complete_train(base, SEED, policy):
                    status(args, 'training', job=number, policy=policy)
                    p.run(args, [sys.executable, '-m', 'e2e_map.experiment', '--variant', policy,
                        '--seed', SEED, '--episodes', 300000, '--out', base / 'train',
                        '--microbatch', args.microbatch, '--device', args.device, '--resume'], base / 'training.log')
                if not p.complete_train(base, SEED, policy):
                    raise ValueError('Selected training did not finish')
            status(args, 'evaluating_matched_control')
            evaluate_control(args)
            for policy in POLICIES:
                base = args.out / f'seed_{SEED}' / policy
                if not p.verify_current(base):
                    status(args, 'evaluating', policy=policy)
                    p.run(args, [sys.executable, '-m', 'e2e_map.evaluate', '--checkpoint', base / 'train/last.pt',
                        '--out', base / 'evaluation', '--device', args.device], base / 'evaluation.log')
                    p.run(args, [sys.executable, '-m', 'ibe_map.verify_results', '--out', base / 'evaluation'], base / 'verification.log')
            status(args, 'summarizing')
            summarize(args)
            if args.publish:
                status(args, 'publishing')
                publish(args)
            status(args, 'complete', published=args.publish)
        except BaseException as exc:
            status(args, 'failed', error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=p.ROOT / 'output/e2e_multiseed')
    parser.add_argument('--frozen', type=Path, default=p.ROOT / 'output/ibe_multiseed')
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--follow-pid', type=int)
    parser.add_argument('--publish', action='store_true')
    execute(parser.parse_args())
