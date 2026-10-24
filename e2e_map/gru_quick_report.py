"""Audit and summarize a small matched GRU experiment (no statistical inference)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Required valid JSON is missing at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    return out


def _checkpoint(path: Path) -> dict[str, Any]:
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint payload is not an object: {path}")
    state = value.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint lacks state_dict: {path}")
    return value


def _delta(before: Any, after: Any, label: str) -> float:
    import torch
    if not torch.is_tensor(before) or not torch.is_tensor(after) or before.shape != after.shape:
        raise ValueError(f"Missing or incompatible checkpoint tensor {label}")
    diff = (after.detach().cpu().float() - before.detach().cpu().float()).norm().item()
    if not math.isfinite(diff):
        raise ValueError(f"Non-finite parameter change for {label}")
    return float(diff)


def _tensor_sha(value: Any, label: str) -> str:
    import torch
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise ValueError(f"Missing or non-finite tensor {label}")
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


GRU_NAMES = ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")
MATCH_KEYS = ("seed", "requested_episodes", "training_scenes", "validation_scenes",
              "protocol", "learning_rate", "optimizer", "batch_size", "environments",
              "replay_capacity", "learning_starts", "updates_per_vector_step",
              "target_update_every", "epsilon", "double_dqn", "loss",
              "gradient_clip_norm", "recurrent", "checkpoint_selection")
SPLITS = ("simple", "hard", "hard_fixed_3_moves")
METRICS = ("accuracy", "return_mean", "distance_m")


def _load_run(root: Path, seed: int, group: str, variant: str,
              episodes: int) -> dict[str, Any]:
    base = root / f"seed_{seed}" / group
    train = base / "train"
    evaluation = base / "evaluation"
    training_path, results_path = train / "training.json", evaluation / "results.json"
    verification_path = evaluation / "verification.json"
    initial_path = train / "snapshots" / "episodes_000000.pt"
    checkpoint_path = train / "last.pt"
    training, results, verification = (_json(training_path), _json(results_path),
                                       _json(verification_path))
    config = training.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{training_path}: missing config")
    completed = training.get("completed_episodes")
    envs = config.get("environments", 64)
    if (isinstance(completed, bool) or not isinstance(completed, int) or
            completed < episodes or completed > episodes + envs - 1):
        raise ValueError(f"{training_path}: completed episodes {completed!r} outside [{episodes}, {episodes + envs - 1}]")
    if config.get("requested_episodes") != episodes or config.get("seed") != seed:
        raise ValueError(f"{training_path}: seed or requested episode budget mismatch")
    if config.get("phase") != "full" or training.get("complete", True) is not True:
        raise ValueError(f"{training_path}: short run is incomplete or not full-scene")
    if config.get("variant") != variant:
        raise ValueError(f"{training_path}: variant mismatch")
    for name in ("updates",):
        if isinstance(training.get(name), bool) or not isinstance(training.get(name), int) or training[name] < 1:
            raise ValueError(f"{training_path}: missing or invalid {name}")
    elapsed = _finite(training.get("elapsed_seconds"), f"{training_path}:elapsed_seconds")
    if elapsed <= 0:
        raise ValueError(f"{training_path}: elapsed_seconds must be positive")
    if group == "e2e":
        if training.get("amp_skipped_updates") != 0:
            raise ValueError(f"{training_path}: expected zero AMP skipped updates")
        if training.get("batchnorm_running_stats_unchanged") is not True:
            raise ValueError(f"{training_path}: BatchNorm running statistics were not verified unchanged")
    for path in (initial_path, checkpoint_path):
        if not path.is_file():
            raise ValueError(f"Required checkpoint is missing: {path}")
    import torch
    torch.set_num_threads(2)
    before_payload, after_payload = _checkpoint(initial_path), _checkpoint(checkpoint_path)
    before, after = before_payload["state_dict"], after_payload["state_dict"]
    if before_payload.get("completed_episodes") != 0:
        raise ValueError(f"{initial_path}: initial snapshot must have completed_episodes=0")
    if before_payload.get("config") != config or after_payload.get("config") != config:
        raise ValueError(f"{checkpoint_path}: checkpoint config differs from training.json")
    if (after_payload.get("completed_episodes") != completed or
            after_payload.get("updates") != training["updates"]):
        raise ValueError(f"{checkpoint_path}: checkpoint counters differ from training.json")
    prefix = "head.gru." if group == "e2e" else "gru."
    gru_delta, gru_initial_sha256 = {}, {}
    for name in GRU_NAMES:
        key = prefix + name
        delta = _delta(before.get(key), after.get(key), key)
        if delta <= 0:
            raise ValueError(f"GRU tensor did not change: {key}")
        gru_delta[name] = delta
        gru_initial_sha256[name] = _tensor_sha(before.get(key), key)
    backbone_delta = {}
    if group == "e2e":
        precision = str(config.get("precision", "")).lower()
        if "bfloat16" not in precision:
            raise ValueError(f"{training_path}: E2E precision does not declare bfloat16")
        for stage in ("conv1", "layer1", "layer2", "layer3", "layer4"):
            keys = [key for key in before if key.startswith(f"encoder.{stage}.")
                    and key in after and key.endswith("weight")]
            changed = [_delta(before[key], after[key], key) for key in keys]
            if not changed or not any(d > 0 for d in changed):
                raise ValueError(f"E2E backbone stage did not change: {stage}")
            backbone_delta[stage] = max(changed)
        bn_keys = [k for k in before if k.startswith("encoder.") and
                   ("running_mean" in k or "running_var" in k or "num_batches_tracked" in k)]
        if not bn_keys or any(not torch.equal(before[k], after.get(k)) for k in bn_keys):
            raise ValueError(f"{training_path}: BatchNorm running statistics changed")
    else:
        signature = config.get("observation_signature")
        preprocess = signature.get("preprocess") if isinstance(signature, dict) else None
        if not isinstance(preprocess, dict):
            raise ValueError(f"{training_path}: frozen observation preprocess signature is missing")
        if "bfloat16" not in str(preprocess.get("encoder_compute_precision", "")).lower():
            raise ValueError(f"{training_path}: frozen encoder precision must be bfloat16")
        if "float32" not in str(preprocess.get("features", "")).lower():
            raise ValueError(f"{training_path}: frozen feature boundary must be float32")

    metrics = results.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError(f"{results_path}: metrics must be a list")
    index = {}
    for row in metrics:
        if isinstance(row, dict) and row.get("policy") == variant:
            key = row.get("split")
            if key in index:
                raise ValueError(f"{results_path}: duplicate metric split {key}")
            index[key] = row
    for split in SPLITS:
        row = index.get(split)
        if row is None:
            raise ValueError(f"{results_path}: missing {split}/{variant} metrics")
        for metric in METRICS:
            _finite(row.get(metric), f"{results_path}:{split}:{metric}")
        if row.get("episodes") != 15125:
            raise ValueError(f"{results_path}: {split} expected 15125 full-scene episodes")
    if results.get("protocol", {}).get("evaluation_seed") != seed:
        raise ValueError(f"{results_path}: evaluation seed mismatch")
    if results["protocol"].get("checkpoint_sha256", {}).get(variant) != _sha(checkpoint_path):
        raise ValueError(f"{results_path}: evaluated checkpoint hash mismatch")
    if verification.get("passed") is not True:
        raise ValueError(f"{verification_path}: independent verification did not pass")
    verify_hashes = verification.get("input_sha256")
    if not isinstance(verify_hashes, dict):
        raise ValueError(f"{verification_path}: missing input_sha256")
    verified_files = [results_path, evaluation / "evaluation.csv", evaluation / "trajectories.jsonl.gz"]
    for path in verified_files:
        key = str(path.resolve())
        if not path.is_file() or verify_hashes.get(key) != _sha(path):
            raise ValueError(f"{verification_path}: missing or stale hash for {path.name}")
    counts = verification.get("counts", {})
    for key in ("trajectories", "csv_rows", "unique_episode_keys"):
        if counts.get(key) != 45375:
            raise ValueError(f"{verification_path}: {key} must equal 45375")
    return {"training": training, "config": config, "metrics": index,
            "gru_delta_l2": gru_delta, "backbone_stage_delta_l2": backbone_delta,
            "gru_initial_sha256": gru_initial_sha256,
            "completed_episodes": completed, "updates": training["updates"],
            "elapsed_seconds": elapsed, "verified_counts": counts,
            "input_sha256": {str(p.resolve()): _sha(p) for p in
                             (training_path, initial_path, checkpoint_path, results_path,
                              verification_path, evaluation / "evaluation.csv",
                              evaluation / "trajectories.jsonl.gz")}}


def build_report(out: str | Path, variant: str, seeds: Sequence[int], episodes: int) -> dict[str, Any]:
    """Validate requested short runs and write their compact, per-seed report."""
    if variant not in ("ibe_map", "memo_map"):
        raise ValueError("variant must be ibe_map or memo_map")
    seeds = tuple(int(seed) for seed in seeds)
    if (not 1 <= len(seeds) <= 2 or len(set(seeds)) != len(seeds) or
            any(seed < 0 for seed in seeds) or not 1000 <= episodes <= 10000):
        raise ValueError("seeds must be one or two distinct nonnegative values; episodes must be 1000..10000")
    root = Path(out).resolve()
    project_root = Path(__file__).resolve().parents[1]
    runs, comparisons = [], []
    for seed in seeds:
        e2e = _load_run(root, seed, "e2e", variant, episodes)
        frozen = _load_run(root, seed, "frozen", variant, episodes)
        if e2e["gru_initial_sha256"] != frozen["gru_initial_sha256"]:
            raise ValueError(f"seed {seed}: E2E/frozen initial GRU tensors differ")
        for key in MATCH_KEYS:
            if e2e["config"].get(key) != frozen["config"].get(key):
                raise ValueError(f"seed {seed}: matched training setting config.{key} differs")
        for split in SPLITS:
            comparisons.append({"seed": seed, "split": split,
                "e2e": {m: float(e2e["metrics"][split][m]) for m in METRICS},
                "frozen": {m: float(frozen["metrics"][split][m]) for m in METRICS}})
        runs.append({"seed": seed, "e2e": {k: v for k, v in e2e.items() if k != "metrics" and k != "config"},
                     "frozen": {k: v for k, v in frozen.items() if k != "metrics" and k != "config"},
                     "comparison": [x for x in comparisons if x["seed"] == seed]})
    provenance = {}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        if (manifest.get("variant") != variant or manifest.get("seeds") != list(seeds) or
                manifest.get("episodes") != episodes):
            raise ValueError(f"{manifest_path}: experiment identity differs from report request")
        sources = manifest.get("source_sha256")
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f"{manifest_path}: source_sha256 is missing")
        for source, expected in sources.items():
            source_path = (project_root / source).resolve()
            if project_root not in source_path.parents or not source_path.is_file() or _sha(source_path) != expected:
                raise ValueError(f"{manifest_path}: source hash mismatch for {source}")
        provenance = {"manifest_path": str(manifest_path),
                      "manifest_sha256": _sha(manifest_path),
                      "source_sha256": manifest.get("source_sha256"),
                      "control_cache_manifest_sha256": manifest.get("control_cache_manifest_sha256")}
    summary = {"status": "complete", "experiment": "matched short-budget GRU exploration",
        "variant": variant, "seeds": list(seeds), "requested_episodes_per_run": episodes,
        "run_count": len(runs) * 2, "provenance": provenance, "runs": runs,
        "interpretation_limits": [
            "仅为1–2个seed的短预算探索，不计算跨seed置信区间或显著性，也不构成完整论文复现。",
            f"训练预算为每组请求{episodes:,}回合；探索率日程按短预算压缩。full表示训练场景全集，不表示300k回合。",
            "SAM为预训练权重，未在UEVAVD上微调。",
            "旧MAP 300k结果与本短预算不构成公平对照。"]}
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    reference = Path(__file__).resolve().parents[1] / "docs" / "gru_quick_reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# GRU quick experiment", "", f"Variant: `{variant}`; requested budget: {episodes} episodes per run.",
             "", "| Seed | Split | E2E accuracy | Frozen accuracy | E2E return | Frozen return | E2E distance (m) | Frozen distance (m) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for item in comparisons:
        a, b = item["e2e"], item["frozen"]
        lines.append(f"| {item['seed']} | {item['split']} | {a['accuracy']:.4f} | {b['accuracy']:.4f} | {a['return_mean']:.4f} | {b['return_mean']:.4f} | {a['distance_m']:.2f} | {b['distance_m']:.2f} |")
    lines.extend(["", "## Training and parameter audit", "", "Per-run actual episodes, updates, elapsed time, GRU tensor deltas, input hashes, and verification counts are in [metrics.json](metrics.json).", "", "## Limits", ""])
    lines.extend(f"- {item}" for item in summary["interpretation_limits"])
    (reference / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant", choices=("ibe_map", "memo_map"), default="ibe_map")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(build_report(args.out, args.variant, args.seeds, args.episodes),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
