"""Finish and report only the already-running seed-42 end-to-end MAP job.

This bounded workflow never trains a model or evaluates SAM/GRU. It waits for
the user-selected process, validates the full-budget MAP checkpoint, evaluates
that MAP model and its same-seed BF16 frozen control, and writes a descriptive
single-seed reference report without confidence intervals.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import pipeline as p
from .report import _load_e2e, read_json

SEED = 42
POLICY = "map"
EPISODES = 300_000
MAX_OVERSHOOT = 63
EXPECTED_EVALUATION_EPISODES = 105_875
SCOPE_SOURCES = (
    "e2e_map/finish_current.py",
    "e2e_map/experiment.py", "e2e_map/model.py", "e2e_map/images.py", "e2e_map/evaluate.py",
    "e2e_map/control.py", "map_baseline/environment.py", "map_baseline/model.py",
    "map_baseline/experiment.py", "map_baseline/features.py", "map_baseline/checkpoints.py",
    "ibe_map/model.py", "ibe_map/experiment.py", "ibe_map/verify_results.py",
    "e2e_map/report.py", "e2e_map/pipeline.py", "ibe_map/multiseed.py",
    "e2e_map/test_finish_current.py",
)
METRICS = ("accuracy", "return_mean", "distance_m", "decisions", "success_rate",
           "stop_rate", "budget_end_rate")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wait_existing(args) -> None:
    """Wait for the explicitly selected Windows process; never terminate it."""
    if not args.follow_pid:
        return
    if os.name != "nt":
        raise ValueError("--follow-pid is supported only on Windows")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, args.follow_pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # It has already exited; the full checkpoint is audited next.
            return
        raise OSError(error, "Cannot attach to the existing MAP training process")
    try:
        status(args, "training", existing_training_pid=args.follow_pid)
        while True:
            result = kernel.WaitForSingleObject(handle, 10_000)
            if result == 0:
                return
            if result != 258:
                raise OSError("Unexpected wait status for existing MAP training process")
    finally:
        kernel.CloseHandle(handle)


def status(args, stage: str, **fields) -> None:
    value = {"stage": stage, "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
             "pid": os.getpid(), "mode": "map_only", "seeds": [SEED],
             "e2e_training_jobs": 1, "cancelled_e2e_ibe_jobs": 1,
             "cancelled_e2e_jobs": 9, "performance_test_pending": True,
             **fields}
    p.write_json(args.out / "status.json", value)
    print(json.dumps(value), flush=True)


def scope_identity(out: Path) -> dict[str, Any]:
    """Lock the reduced scope to the original plan and the current source set."""
    plan_path = out / "run_manifest.json"
    if not plan_path.is_file():
        raise ValueError(f"Original full experiment manifest is missing: {plan_path}")
    plan = read_json(plan_path)
    if (plan.get("seeds") != [42, 43, 44, 45, 46] or
            plan.get("variants") != ["map", "ibe_map"] or plan.get("episodes") != EPISODES):
        raise ValueError("Original manifest does not match the five-seed 300k plan")
    source_lock = plan.get("source_sha256")
    expected_sources = set(p.TRAIN_SOURCES)
    if not isinstance(source_lock, dict) or not expected_sources.issubset(source_lock):
        raise ValueError("Original manifest is missing its complete training source lock")
    changed = [name for name in p.TRAIN_SOURCES
               if sha256(p.ROOT / name) != source_lock[name]]
    if changed:
        raise ValueError(f"Training source lock changed after launch: {', '.join(changed)}")
    identity = {
        "schema_version": 1,
        "mode": "map_only_single_seed_reference",
        "seed": SEED,
        "policy": POLICY,
        "requested_training_episodes": EPISODES,
        "original_run_manifest_sha256": sha256(plan_path),
        "original_training_source_sha256": source_lock,
        "evaluation_source_sha256": {name: sha256(p.ROOT / name) for name in SCOPE_SOURCES},
        "scope": "finish current seed-42 E2E MAP and compare with same-seed BF16 frozen MAP; no E2E IBE-MAP/SAM+GRU run",
    }
    scope_path = out / "scope_manifest.json"
    if scope_path.is_file() and read_json(scope_path) != identity:
        raise ValueError("MAP-only scope/source identity changed; refusing to reuse this output")
    if not scope_path.is_file():
        p.write_json(scope_path, identity)
    return identity


def validate_full_training(training: dict[str, Any], seed: int = SEED,
                           variant: str = POLICY, requested: int = EPISODES,
                           plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check the complete final checkpoint record against the original plan."""
    config = training.get("config")
    if not isinstance(config, dict):
        raise ValueError("Training record lacks an auditable config")
    if (config.get("seed") != seed or config.get("variant") != variant or
            config.get("phase") != "full" or config.get("requested_episodes") != requested):
        raise ValueError("Training identity must be full-budget seed-42 MAP")
    if training.get("complete") is not True:
        raise ValueError("Training did not reach its requested full budget")
    completed = training.get("completed_episodes")
    if isinstance(completed, bool) or not isinstance(completed, int) or not requested <= completed <= requested + MAX_OVERSHOOT:
        raise ValueError("Completed episode count is outside the full-budget overshoot tolerance")
    if not isinstance(training.get("updates"), int) or training["updates"] < 1:
        raise ValueError("Training record has no successful optimizer updates")
    changes = training.get("backbone_parameter_l2_change")
    if not isinstance(changes, dict) or not changes or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or value <= 0 for value in changes.values()):
        raise ValueError("Training record does not prove finite backbone updates")
    if training.get("batchnorm_running_stats_unchanged") is not True:
        raise ValueError("BatchNorm running-stat identity check failed")
    if plan is not None:
        if config.get("training_scenes") != plan.get("training_scenes"):
            raise ValueError("Training scenes differ from the original run manifest")
        if config.get("validation_scenes") != []:
            raise ValueError("Full MAP run must not use validation scenes")
        if config.get("protocol") != plan.get("protocol"):
            raise ValueError("Training protocol differs from the original run manifest")
    if config.get("precision") != (
            "FP32 weights and policy heads; CUDA bfloat16 encoder autocast; FP32 feature boundary; GradScaler initial128"):
        raise ValueError("Training config does not declare the audited BF16/FP32 precision boundary")
    return config


def _verify_raw_control_cache(control_root: Path, control_module) -> dict[str, Any]:
    """Verify just the raw BF16 control cache needed for MAP evaluation."""
    manifest_path = control_root / "cache" / "control_cache_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("scenes") != sorted(control_module.TRAIN + control_module.SIMPLE + control_module.HARD):
        raise ValueError("Matched frozen raw cache does not cover the fixed evaluation grid")
    if manifest.get("source_code_sha256") != control_module._code_hashes():
        raise ValueError("Matched frozen cache source code hash differs")
    if manifest.get("encoder_code_sha256") != control_module._combined_hash(
            control_module._code_hashes(control_module.ENCODER_SOURCE_FILES)):
        raise ValueError("Matched frozen encoder source hash differs")
    if sha256(control_module.WEIGHTS) != manifest.get("encoder_weights_sha256"):
        raise ValueError("Matched frozen ResNet50 weights differ from the cache manifest")

    from map_baseline.environment import Dataset
    scenes = sorted(control_module.TRAIN + control_module.SIMPLE + control_module.HARD)
    source_cache = control_module.RAW_SOURCE
    target_cache = control_root / "cache" / "raw"
    source_data = Dataset(source_cache, scenes)
    if source_data.observation_signature != manifest.get("raw_observation_signature"):
        raise ValueError("Original raw observation cache identity differs")
    for scene in scenes:
        key = f"raw/{scene:03d}"
        source_npz, source_meta = control_module._scene_paths(source_cache, scene)
        target_npz, target_meta = control_module._scene_paths(target_cache, scene)
        source_entry = manifest.get("source_caches", {}).get(key)
        target_entry = manifest.get("control_caches", {}).get("raw", {}).get(f"{scene:03d}")
        if not source_entry or not target_entry:
            raise ValueError(f"Raw cache manifest is missing scene {scene:03d}")
        if (sha256(source_npz) != source_entry.get("npz_sha256") or
                sha256(source_meta) != source_entry.get("metadata_sha256")):
            raise ValueError(f"Source raw cache changed: {source_npz}")
        target_info = read_json(target_meta)
        if (sha256(target_npz) != target_info.get("npz_sha256") or
                sha256(target_npz) != target_entry.get("npz_sha256") or
                sha256(target_meta) != target_entry.get("metadata_sha256") or
                target_info.get("source_npz_sha256") != source_entry.get("npz_sha256") or
                target_info.get("source_metadata_sha256") != source_entry.get("metadata_sha256") or
                target_info.get("encoder_weights_sha256") != manifest.get("encoder_weights_sha256") or
                target_info.get("encoder_code_sha256") != manifest.get("encoder_code_sha256")):
            raise ValueError(f"Matched BF16 raw cache hash mismatch: {target_npz}")
        expected_signature = control_module._control_signature(
            source_entry["signature"], "CUDA bfloat16 autocast")
        if target_info.get("signature") != expected_signature:
            raise ValueError(f"Matched BF16 raw cache signature mismatch: {target_meta}")
        import numpy as np
        with np.load(source_npz, allow_pickle=False) as source, np.load(target_npz, allow_pickle=False) as target:
            for key_name in ("probs", "boxes", "visible", "positions", "names"):
                if not np.array_equal(source[key_name], target[key_name]):
                    raise ValueError(f"Frozen raw cache changed immutable {key_name}: {target_npz}")
            if target["features"].dtype != np.float32 or target["features"].shape != (605, 2048):
                raise ValueError(f"Frozen raw cache has invalid feature storage: {target_npz}")
            if not np.isfinite(target["features"]).all():
                raise ValueError(f"Frozen raw cache has non-finite features: {target_npz}")
    return manifest


def _run(command: list[Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nRUN " + json.dumps([str(item) for item in command]) + "\n")
        log.flush()
        subprocess.run([str(item) for item in command], cwd=p.ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})


def _verified_evaluation(evaluation: Path, checkpoint: Path, verifier_sha: str,
                        require_protocol_splits: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path, verification_path = evaluation / "results.json", evaluation / "verification.json"
    results, verification = read_json(result_path), read_json(verification_path)
    protocol = results.get("protocol", {})
    if protocol.get("evaluation_seed") != SEED:
        raise ValueError(f"Evaluation seed mismatch: {result_path}")
    if require_protocol_splits and protocol.get("evaluation_splits") != ["train", "simple", "hard"]:
        raise ValueError(f"Evaluation must cover all three native splits: {result_path}")
    if protocol.get("checkpoint_sha256", {}).get(POLICY) != sha256(checkpoint):
        raise ValueError(f"Evaluation checkpoint does not match {checkpoint}")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise ValueError(f"Independent trajectory verification did not pass: {verification_path}")
    counts = verification.get("counts", {})
    for key in ("trajectories", "csv_rows", "unique_episode_keys"):
        if counts.get(key) != EXPECTED_EVALUATION_EPISODES:
            raise ValueError(f"Independent verifier {key} count is incomplete")
    inputs = verification.get("input_sha256")
    if not isinstance(inputs, dict):
        raise ValueError("Independent verification has no input hash map")
    for name in ("results.json", "evaluation.csv", "trajectories.jsonl.gz"):
        path = (evaluation / name).resolve()
        if inputs.get(str(path)) != sha256(path):
            raise ValueError(f"Independent verification does not cover {name}")
    if verification.get("verifier_sha256") and verification["verifier_sha256"] != verifier_sha:
        raise ValueError("Independent verification was produced by a different verifier source")
    return results, verification


def _metric_groups(results: dict[str, Any], variant: str) -> dict[str, dict[str, Any]]:
    rows = results.get("metrics")
    if not isinstance(rows, list):
        raise ValueError("Evaluation metrics must be a list")
    groups = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("policy") != variant:
            continue
        split = row.get("split")
        if split in groups:
            raise ValueError(f"Duplicate evaluation metric group: {split}/{variant}")
        groups[split] = row
    for split in ("train", "simple", "hard", "hard_fixed_3_moves"):
        if split not in groups:
            raise ValueError(f"Evaluation is missing {split}/{variant}")
        for metric in METRICS:
            value = groups[split].get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid {split}/{variant} metric {metric}")
    return groups


def build_result(e2e_results: dict[str, Any], frozen_results: dict[str, Any],
                 e2e_training: dict[str, Any], frozen_training: dict[str, Any],
                 e2e_verification: dict[str, Any], frozen_verification: dict[str, Any],
                 scope: dict[str, Any], e2e_paths: dict[str, Path], frozen_paths: dict[str, Path]) -> dict[str, Any]:
    e2e_groups = _metric_groups(e2e_results, POLICY)
    frozen_groups = _metric_groups(frozen_results, POLICY)
    rows = {}
    for split in ("train", "simple", "hard", "hard_fixed_3_moves"):
        e_row, f_row = e2e_groups[split], frozen_groups[split]
        if e_row["episodes"] != f_row["episodes"]:
            raise ValueError(f"E2E/frozen {split} episode counts differ")
        rows[split] = {
            "e2e_map": {key: e_row[key] for key in ("episodes", *METRICS)},
            "frozen_bf16_map": {key: f_row[key] for key in ("episodes", *METRICS)},
            "e2e_minus_frozen": {key: float(e_row[key]) - float(f_row[key]) for key in METRICS},
        }
    return {
        "schema_version": 1,
        "status": "complete_single_seed_map_reference",
        "scope": "seed-42 MAP only; no E2E IBE-MAP/SAM+GRU training or evaluation",
        "seed": SEED,
        "requested_training_episodes": EPISODES,
        "hard_evaluation_episodes_per_group": 15_125,
        "verified_trajectories_each_run": EXPECTED_EVALUATION_EPISODES,
        "verified_trajectories_total": EXPECTED_EVALUATION_EPISODES * 2,
        "groups": rows,
        "uncertainty": None,
        "statistical_claims": None,
        "sam_gru_conclusion": "Not evaluated; these MAP-only results cannot determine whether SAM+GRU improves with end-to-end encoder training.",
        "training_runs": [
            _training_record("e2e_map", e2e_training, e2e_paths),
            _training_record("frozen_bf16_map", frozen_training, frozen_paths),
        ],
        "verification": {
            "e2e_map": {"passed": True, "counts": e2e_verification["counts"],
                         "verification_sha256": sha256(e2e_paths["verification"])},
            "frozen_bf16_map": {"passed": True, "counts": frozen_verification["counts"],
                                "verification_sha256": sha256(frozen_paths["verification"])},
        },
        "source_lock": {
            "original_run_manifest_sha256": scope["original_run_manifest_sha256"],
            "original_training_source_sha256": scope["original_training_source_sha256"],
            "evaluation_source_sha256": scope["evaluation_source_sha256"],
            "e2e_training_json_sha256": sha256(e2e_paths["training"]),
            "e2e_checkpoint_sha256": sha256(e2e_paths["checkpoint"]),
            "frozen_training_json_sha256": sha256(frozen_paths["training"]),
            "frozen_checkpoint_sha256": sha256(frozen_paths["checkpoint"]),
        },
        "limitations": [
            "One E2E MAP seed only: no across-seed standard deviation, confidence interval, or significance claim.",
            "No E2E IBE-MAP/SAM+GRU run was completed; this report cannot answer whether SAM+GRU benefits from end-to-end training.",
            "The primary same-seed comparison changes the ResNet50 training regime; it does not isolate SAM or GRU.",
            "SAM remains at official pretrained weights and is not fine-tuned on UEVAVD.",
        ],
    }


def _training_record(label: str, training: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "regime": label,
        "seed": SEED,
        "requested_episodes": EPISODES,
        "completed_episodes": training["completed_episodes"],
        "updates": training.get("updates"),
        "elapsed_seconds": training.get("elapsed_seconds"),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "training_json_sha256": sha256(paths["training"]),
        "config_sha256": _canonical_sha256(training["config"]),
    }


def write_readme(result: dict[str, Any], destination: Path) -> None:
    def table(split: str, title: str) -> str:
        values = result["groups"][split]
        lines = [f"### {title}", "", "| 组别 | 准确率（%） | 平均折扣回报 | 平均移动距离（m） |", "|---|---:|---:|---:|"]
        for key, label in (("frozen_bf16_map", "同 seed BF16 冻结 MAP"), ("e2e_map", "端到端 MAP")):
            row = values[key]
            lines.append(f"| {label} | {100*row['accuracy']:.2f} | {row['return_mean']:.4f} | {row['distance_m']:.2f} |")
        delta = values["e2e_minus_frozen"]
        lines.append(f"| 端到端 − 冻结（描述性差值） | {100*delta['accuracy']:+.2f} pp | {delta['return_mean']:+.4f} | {delta['distance_m']:+.2f} |")
        return "\n".join(lines)

    tr, fr = result["training_runs"]
    document = f'''# ResNet50 端到端 MAP：单种子参考

本报告只包含 seed 42 的 MAP：当前端到端 MAP 与同 seed、同 BF16 编码器精度的冻结 MAP。两个训练任务各请求 {EPISODES:,} 回合；端到端任务实际完成 {tr['completed_episodes']:,} 回合，冻结任务实际完成 {fr['completed_episodes']:,} 回合。所有评估 split 均使用固定的全部起点，并通过独立轨迹核验。

**只有一个训练 seed，因此这里只报告逐组原始值和端到端减冻结的描述性差值；不报告跨 seed SD、置信区间、显著性或稳定提升结论。没有运行 E2E IBE-MAP/SAM+GRU，不能根据 MAP-only 结果判断 SAM+GRU 的变化。**

{table('train', 'Train split')}

{table('simple', 'Simple split')}

{table('hard', 'Hard split')}

{table('hard_fixed_3_moves', 'Hard split：固定三步诊断')}

每个方法各有 {result['verified_trajectories_each_run']:,} 条经独立核验的轨迹，合计 {result['verified_trajectories_total']:,} 条。训练 checkpoint、配置和核验文件的 SHA-256、更新数、训练时长及实际回合数见 [`metrics.json`](metrics.json)。完整身份记录见输出目录的 `scope_manifest.json`；旧五种子计划及其源哈希保持为 provenance。

SAM 使用官方预训练权重，未在 UEVAVD 上微调。本文仅比较 MAP，不作 SAM+GRU 结论，也不把 encoder 可训练性变化解释为 SAM 或 GRU 的独立效果。

## English brief

This is a seed-42 MAP-only reference: end-to-end ResNet50 MAP versus the same-seed BF16 frozen MAP control. Each run requested 300,000 episodes and both were independently verified on all evaluation splits. With one seed, the report gives descriptive values and paired differences only; it makes no confidence-interval or significance claim. E2E IBE-MAP/SAM+GRU was not run, so these MAP results cannot answer whether SAM+GRU benefits from end-to-end training.
'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def _verify_and_run(args) -> dict[str, Any]:
    from . import control as c

    scope = scope_identity(args.out)
    plan = read_json(args.out / "run_manifest.json")
    e2e_base = args.out / "seed_42" / "map"
    e2e_training_path, e2e_checkpoint = e2e_base / "train/training.json", e2e_base / "train/last.pt"
    e2e_training = read_json(e2e_training_path)
    e2e_config = validate_full_training(e2e_training, plan=plan)
    if not e2e_checkpoint.is_file():
        raise ValueError("Completed E2E MAP checkpoint is missing")

    control_root = args.out / "control"
    control_cache_manifest = _verify_raw_control_cache(control_root, c)
    frozen_base = control_root / "seed_42" / "map"
    frozen_cache = control_root / "cache" / "raw"
    if not c._complete_job(frozen_base, SEED, POLICY, frozen_cache, None):
        raise ValueError("Same-seed BF16 frozen MAP training is incomplete or mismatched")
    frozen_training_path, frozen_checkpoint = frozen_base / "train/training.json", frozen_base / "train/last.pt"
    frozen_training = read_json(frozen_training_path)
    frozen_config = frozen_training["config"]
    frozen_completed = frozen_training.get("completed_episodes")
    if (isinstance(frozen_completed, bool) or not isinstance(frozen_completed, int) or
            not EPISODES <= frozen_completed <= EPISODES + MAX_OVERSHOOT or
            frozen_training.get("updates", 0) < 1):
        raise ValueError("Same-seed BF16 frozen MAP did not complete 300,000 episodes")

    from .report import _validate_match
    e2e_eval = e2e_base / "evaluation"
    status(args, "evaluating_e2e_map")
    _run([sys.executable, "-m", "e2e_map.evaluate", "--checkpoint", e2e_checkpoint,
          "--out", e2e_eval, "--raw-cache", p.ROOT / "output/map/cache",
          "--splits", "train", "simple", "hard", "--device", args.device],
         e2e_base / "evaluation.log")
    _run([sys.executable, "-m", "ibe_map.verify_results", "--out", e2e_eval,
          "--raw-cache", p.ROOT / "output/map/cache", "--data", p.ROOT / "dataset"],
         e2e_base / "verification.log")

    frozen_eval = control_root / "seed_42" / "map_only_evaluation"
    status(args, "evaluating_frozen_map")
    frozen_eval.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "ibe_map.experiment", "evaluate", "--policies", "map",
          "--raw-cache", frozen_cache, "--map-checkpoint", frozen_checkpoint,
          "--out", frozen_eval, "--seed", SEED, "--device", args.device, "--no-report"],
         control_root / "seed_42" / "map_only_evaluation.log")
    _run([sys.executable, "-m", "ibe_map.verify_results", "--out", frozen_eval,
          "--raw-cache", frozen_cache, "--data", p.ROOT / "dataset"],
         control_root / "seed_42" / "map_only_verification.log")

    e2e_results, e2e_verification = _verified_evaluation(e2e_eval, e2e_checkpoint,
                                                         sha256(p.ROOT / "ibe_map/verify_results.py"))
    frozen_results, frozen_verification = _verified_evaluation(
        frozen_eval, frozen_checkpoint, sha256(p.ROOT / "ibe_map/verify_results.py"),
        require_protocol_splits=False)
    _validate_match(frozen_config, e2e_config, e2e_results, SEED, POLICY)
    # The full frozen verifier output is kept outside the dual-policy schema.
    frozen_paths = {"training": frozen_training_path, "checkpoint": frozen_checkpoint,
                    "verification": frozen_eval / "verification.json"}
    e2e_paths = {"training": e2e_training_path, "checkpoint": e2e_checkpoint,
                 "verification": e2e_base / "evaluation/verification.json"}
    result = build_result(e2e_results, frozen_results, e2e_training, frozen_training,
                          e2e_verification, frozen_verification, scope,
                          e2e_paths, frozen_paths)
    result["source_lock"]["frozen_control_cache_manifest_sha256"] = sha256(
        control_root / "cache/control_cache_manifest.json")
    result["source_lock"]["frozen_encoder_weights_sha256"] = control_cache_manifest["encoder_weights_sha256"]
    result_path = args.out / "map_only_summary.json"
    p.write_json(result_path, result)
    verification_paths = [result_path, args.out / "scope_manifest.json",
        e2e_paths["training"], e2e_paths["checkpoint"], e2e_paths["verification"],
        frozen_paths["training"], frozen_paths["checkpoint"], frozen_paths["verification"],
        control_root / "cache/control_cache_manifest.json"]
    p.write_json(args.out / "map_only_verification.json", {
        "passed": True,
        "input_sha256": {str(path.resolve()): sha256(path) for path in verification_paths},
    })
    destination = p.ROOT / "docs/e2e_map_reference"
    destination.mkdir(parents=True, exist_ok=True)
    p.write_json(destination / "metrics.json", result)
    write_readme(result, destination / "README.md")
    return result


def execute(args) -> dict[str, Any]:
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with p.single_process(args.out / "pipeline.lock"), p.keep_awake():
        try:
            status(args, "waiting")
            wait_existing(args)
            status(args, "validating")
            result = _verify_and_run(args)
            status(args, "complete", result="docs/e2e_map_reference", verified_trajectories=result["verified_trajectories_total"])
            return result
        except BaseException as exc:
            status(args, "failed", error=str(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow-pid", type=int, help="wait for the already-running MAP trainer")
    parser.add_argument("--out", type=Path, default=p.ROOT / "output/e2e_multiseed")
    parser.add_argument("--device", default="cuda")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
