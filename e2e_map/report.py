"""Build a matched-seed comparison of frozen and end-to-end ResNet50 runs.

The formal comparison is emitted only when every requested seed and policy has
both a complete training record and a final evaluation. Partial run directories
are useful for monitoring, but are never summarized as final results.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_VARIANTS = ("map", "ibe_map")
METRICS = ("accuracy", "return_mean", "distance_m", "decisions",
           "success_rate", "stop_rate", "budget_end_rate",
           "generalization_gap_return")
T975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105,
        5: 2.570581836, 6: 2.446911851, 7: 2.364624252, 8: 2.306004135,
        9: 2.262157163}
REQUIRED_METRICS = ("accuracy", "return_mean", "distance_m", "decisions",
                    "success_rate", "stop_rate", "budget_end_rate")
PROTOCOL_KEYS = ("max_decisions", "gamma", "terminal_reward", "step_penalty",
                 "distance_penalty_per_unit", "recognition_threshold", "success_mode",
                 "heuristic_stop_threshold")
MATCHED_TRAINING_KEYS = ("phase", "seed", "requested_episodes", "training_scenes",
                         "validation_scenes", "protocol", "learning_rate", "optimizer",
                         "batch_size", "environments", "replay_capacity", "learning_starts",
                         "updates_per_vector_step", "target_update_every", "epsilon",
                         "double_dqn", "loss", "gradient_clip_norm", "recurrent",
                         "checkpoint_selection")


def read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required experiment file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_stats(values: list[float]) -> dict[str, Any]:
    """Sample statistics and a two-sided 95% Student-t interval, without NumPy."""
    vals = [float(value) for value in values]
    if not vals or not all(math.isfinite(value) for value in vals):
        raise ValueError("Statistics require finite, nonempty values")
    n = len(vals)
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "std": None, "ci95_t": None, "n": 1,
                "values": vals, "ddof": None}
    if n - 1 not in T975:
        raise ValueError("Student-t intervals support at most 10 matched seeds")
    variance = sum((value - mean) ** 2 for value in vals) / (n - 1)
    std = math.sqrt(variance)
    margin = T975[n - 1] * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95_t": [mean - margin, mean + margin],
            "n": n, "values": vals, "ddof": 1}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _metric_index(metrics: Any, seed: int, variant: str, source: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(metrics, list):
        raise ValueError(f"{source}: metrics must be a list")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metrics:
        if not isinstance(row, dict) or "split" not in row or "policy" not in row:
            raise ValueError(f"{source}: malformed metric row")
        key = (str(row["split"]), str(row["policy"]))
        if key in index:
            raise ValueError(f"{source}: duplicate metric row {key}")
        index[key] = row
    for split in ("train", "hard"):
        row = index.get((split, variant))
        if row is None:
            raise ValueError(f"{source}: missing {split}/{variant} metric for seed {seed}")
        if (isinstance(row.get("episodes"), bool) or
                not isinstance(row.get("episodes"), int) or row["episodes"] < 1):
            raise ValueError(f"{source}: {split}/{variant} episode count is missing")
        for metric in REQUIRED_METRICS:
            _finite_number(row.get(metric), f"{source}:{split}/{metric}")
        for metric in ("accuracy", "success_rate", "stop_rate", "budget_end_rate"):
            value = _finite_number(row[metric], f"{source}:{split}/{metric}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{source}:{split}/{metric} must be in [0, 1]")
        for metric in ("distance_m", "decisions"):
            if _finite_number(row[metric], f"{source}:{split}/{metric}") < 0:
                raise ValueError(f"{source}:{split}/{metric} must be nonnegative")
    return index


def _load_frozen(summary: dict[str, Any], seed: int, variant: str) -> dict[str, Any]:
    runs = summary.get("final_runs")
    if not isinstance(runs, list):
        raise ValueError("Frozen summary has no final_runs list")
    matches = [run for run in runs if isinstance(run, dict) and run.get("seed") == seed]
    if len(matches) != 1:
        raise ValueError(f"Frozen summary must contain exactly one run for seed {seed}")
    run = matches[0]
    index = _metric_index(run.get("metrics"), seed, variant, f"frozen seed {seed}")
    training = run.get("training")
    if not isinstance(training, dict) or not isinstance(training.get(variant), dict):
        raise ValueError(f"Frozen summary is missing training config for seed {seed}/{variant}")
    config = training[variant].get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Frozen summary is missing auditable config for seed {seed}/{variant}")
    if "recurrent" not in config:
        # Older MAP configs omit this flag; the policy family defines it.
        config = {**config, "recurrent": variant == "ibe_map"}
    return {"run": run, "metrics": index, "config": config}


def _load_e2e(root: Path, seed: int, variant: str, requested_episodes: int,
              expected_hard_episodes: int) -> dict[str, Any]:
    base = root / f"seed_{seed}" / variant
    results_path = base / "evaluation" / "results.json"
    training_path = base / "train" / "training.json"
    checkpoint_path = base / "train" / "last.pt"
    verification_path = base / "evaluation" / "verification.json"
    results = read_json(results_path)
    training = read_json(training_path)
    verification = read_json(verification_path)
    if not isinstance(results, dict) or not isinstance(training, dict):
        raise ValueError(f"seed {seed}/{variant}: result and training files must be JSON objects")

    config = training.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{training_path}: missing config; run is not auditable")
    checks = {"seed": seed, "variant": variant, "requested_episodes": requested_episodes}
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise ValueError(f"{training_path}: config.{key} must equal {expected!r}")
    if config.get("phase", "full") != "full":
        raise ValueError(f"{training_path}: only full-budget training is comparable")
    completed = training.get("completed_episodes")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < requested_episodes:
        raise ValueError(f"{training_path}: training is incomplete ({completed!r}/{requested_episodes})")
    environments = config.get("environments", 64)
    if isinstance(environments, bool) or not isinstance(environments, int) or environments < 1:
        raise ValueError(f"{training_path}: invalid vector environment count")
    if completed > requested_episodes + environments - 1:
        raise ValueError(f"{training_path}: completed episodes exceed the requested budget tolerance")
    updates = training.get("updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError(f"{training_path}: missing or invalid optimizer update count")

    evaluation_seed = results.get("protocol", {}).get("evaluation_seed") if isinstance(results.get("protocol"), dict) else None
    if evaluation_seed != seed:
        raise ValueError(f"{results_path}: evaluation seed {evaluation_seed} does not match training seed {seed}")
    index = _metric_index(results.get("metrics"), seed, variant, str(results_path))
    for split in ("train", "hard"):
        row = index[(split, variant)]
        expected = expected_hard_episodes if split == "hard" else row["episodes"]
        if split == "hard" and row["episodes"] != expected:
            raise ValueError(f"{results_path}: hard evaluation has {row['episodes']} episodes; expected {expected}")
    if not checkpoint_path.is_file():
        raise ValueError(f"{checkpoint_path}: final E2E checkpoint is missing")
    checkpoint_digest = sha256(checkpoint_path)
    reported_checkpoints = results["protocol"].get("checkpoint_sha256")
    if not isinstance(reported_checkpoints, dict) or reported_checkpoints.get(variant) != checkpoint_digest:
        raise ValueError(f"{results_path}: evaluation checkpoint hash does not match {checkpoint_path}")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise ValueError(f"{verification_path}: trajectory verification did not pass")
    verified_inputs = verification.get("input_sha256")
    if not isinstance(verified_inputs, dict) or verified_inputs.get(str(results_path.resolve())) != sha256(results_path):
        raise ValueError(f"{verification_path}: verification does not cover this results file")
    for filename in ("evaluation.csv", "trajectories.jsonl.gz"):
        source_path = base / "evaluation" / filename
        if (not source_path.is_file() or
                verified_inputs.get(str(source_path.resolve())) != sha256(source_path)):
            raise ValueError(f"{verification_path}: verification does not cover {filename}")
    train_count = index[("train", variant)]["episodes"]
    hard_count = index[("hard", variant)]["episodes"]
    expected_trajectory_count = train_count + 3 * hard_count  # train, simple, hard, hard fixed-three
    counts = verification.get("counts")
    for key in ("trajectories", "csv_rows", "unique_episode_keys"):
        if not isinstance(counts, dict) or counts.get(key) != expected_trajectory_count:
            raise ValueError(f"{verification_path}: {key} does not equal expected {expected_trajectory_count}")
    return {"metrics": index, "training": training, "results": results,
            "results_path": str(results_path.resolve()),
            "training_path": str(training_path.resolve()),
            "results_sha256": sha256(results_path), "training_sha256": sha256(training_path),
            "checkpoint_sha256": checkpoint_digest, "verification": verification,
            "verification_path": str(verification_path.resolve()),
            "verification_sha256": sha256(verification_path)}


def _validate_match(frozen_config: dict[str, Any], e2e_config: dict[str, Any],
                    e2e_results: dict[str, Any], seed: int, variant: str) -> None:
    """Compare matched settings while allowing feature cache precision fields to differ."""
    for key in MATCHED_TRAINING_KEYS:
        if key == "recurrent":
            frozen_value = frozen_config.get(key, variant == "ibe_map")
            e2e_value = e2e_config.get(key, variant == "ibe_map")
        else:
            frozen_value = frozen_config.get(key)
            e2e_value = e2e_config.get(key)
        if frozen_value is None or e2e_value is None:
            raise ValueError(f"Seed {seed}/{variant}: missing matched training field {key}")
        if frozen_value != e2e_value:
            raise ValueError(f"Seed {seed}/{variant}: frozen/e2e config.{key} differs")
    if e2e_config.get("encoder") != (
            "end-to-end ImageNet ResNet50 V1; all backbone convolution and BN affine parameters trainable"):
        raise ValueError(f"Seed {seed}/{variant}: run is not marked as end-to-end backbone training")
    if frozen_config.get("encoder") != "frozen ImageNet ResNet50 V1":
        raise ValueError(f"Seed {seed}/{variant}: matched control must declare a frozen ImageNet ResNet50")
    frozen_weights = frozen_config.get("observation_signature", {}).get("pretrained_sha256")
    if not frozen_weights or e2e_config.get("pretrained_sha256") != frozen_weights:
        raise ValueError(f"Seed {seed}/{variant}: pretrained ResNet50 initialization differs")
    frozen_signature = frozen_config.get("observation_signature")
    e2e_signature = e2e_config.get("observation_signature")
    if not isinstance(frozen_signature, dict) or not isinstance(e2e_signature, dict):
        raise ValueError(f"Seed {seed}/{variant}: observation signature is missing")
    frozen_preprocess = frozen_signature.get("preprocess")
    if not isinstance(frozen_preprocess, dict):
        raise ValueError(f"Seed {seed}/{variant}: frozen preprocess signature is missing")
    if frozen_preprocess.get("encoder_compute_precision") != "CUDA bfloat16 autocast":
        raise ValueError(f"Seed {seed}/{variant}: matched frozen encoder must use CUDA bfloat16 autocast")
    if "float32" not in str(frozen_preprocess.get("features", "")).lower():
        raise ValueError(f"Seed {seed}/{variant}: matched frozen features must be stored as float32")
    expected_precision = "FP32 weights and policy heads; CUDA bfloat16 encoder autocast; FP32 feature boundary; GradScaler initial128"
    if e2e_config.get("precision") != expected_precision:
        raise ValueError(f"Seed {seed}/{variant}: E2E config must explicitly declare bfloat16 precision")

    def normalized_signature(value: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(value))
        preprocess = normalized.get("preprocess")
        if not isinstance(preprocess, dict):
            raise ValueError(f"Seed {seed}/{variant}: preprocess signature is malformed")
        preprocess.pop("encoder_compute_precision", None)
        preprocess.pop("features", None)
        return normalized

    if normalized_signature(frozen_signature) != normalized_signature(e2e_signature):
        raise ValueError(f"Seed {seed}/{variant}: observation signatures differ outside encoder precision/features")
    observation_key = "raw_observation_signature" if variant == "map" else "sam_observation_signature"
    result_signature = e2e_results.get("protocol", {}).get(observation_key)
    if result_signature != e2e_config.get("observation_signature"):
        raise ValueError(f"Seed {seed}/{variant}: evaluated observation source signature differs")
    result_protocol = e2e_results.get("protocol")
    if not isinstance(result_protocol, dict):
        raise ValueError(f"Seed {seed}/{variant}: evaluation protocol is missing")
    expected_evaluation_precision = "CUDA bfloat16 autocast; FP32 features and policy heads"
    if result_protocol.get("encoder_precision") != expected_evaluation_precision:
        raise ValueError(
            f"Seed {seed}/{variant}: evaluation protocol.encoder_precision must equal "
            f"{expected_evaluation_precision!r}")
    for key in PROTOCOL_KEYS:
        if (key not in frozen_config["protocol"] or key not in e2e_config["protocol"] or
                frozen_config["protocol"][key] != e2e_config["protocol"][key] or
                result_protocol.get(key) != frozen_config["protocol"][key]):
            raise ValueError(f"Seed {seed}/{variant}: evaluation/training protocol.{key} differs")


def _row_metric(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    return _finite_number(value, metric)


def _export_metrics(index: dict[tuple[str, str], dict[str, Any]], variant: str) -> dict[str, Any]:
    return {split: index[(split, variant)] for split in ("train", "hard")}


def _summary_block(values_by_metric: dict[str, list[float]]) -> dict[str, Any]:
    return {metric: seed_stats(values) for metric, values in values_by_metric.items()}


def build_summary(frozen_summary_path: str | Path, e2e_root: str | Path,
                  seeds: tuple[int, ...] = DEFAULT_SEEDS,
                  variants: tuple[str, ...] = DEFAULT_VARIANTS,
                  historical_summary_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a complete matched-seed frozen/e2e comparison.

    ``e2e_root`` contains ``seed_<seed>/<variant>/{train,evaluation}``. Raises
    ``ValueError`` on any missing, partial, duplicated, or mismatched run.
    """
    seeds, variants = tuple(seeds), tuple(variants)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty sequence of distinct IDs")
    if variants != DEFAULT_VARIANTS:
        raise ValueError(f"variants must be exactly {DEFAULT_VARIANTS}")
    frozen_path, root = Path(frozen_summary_path), Path(e2e_root)
    frozen = read_json(frozen_path)
    if not isinstance(frozen, dict):
        raise ValueError("Frozen summary must contain a JSON object")
    frozen_seeds = frozen.get("seeds")
    if not isinstance(frozen_seeds, list) or not set(seeds).issubset(set(frozen_seeds)):
        raise ValueError("Requested seeds are not all present in frozen summary")
    protocol = frozen.get("protocol", {})
    requested = protocol.get("requested_training_episodes") if isinstance(protocol, dict) else None
    if not isinstance(requested, int) or requested < 1:
        raise ValueError("Frozen summary does not declare requested_training_episodes")
    frozen_hard = next((row for row in frozen.get("final_summary", [])
                        if row.get("split") == "hard" and row.get("policy") == "map"), None)
    if not isinstance(frozen_hard, dict) or not isinstance(frozen_hard.get("episodes_per_run"), int):
        raise ValueError("Frozen summary does not declare hard evaluation episode count")
    hard_episodes = frozen_hard["episodes_per_run"]
    control_verification = frozen.get("verification_total")
    if not isinstance(control_verification, dict) or control_verification.get("passed") is not True:
        raise ValueError("Matched BF16 frozen summary must include a passing verification_total")
    control_verification_path = frozen_path.parent / "summary_verification.json"
    if not control_verification_path.is_file():
        raise ValueError("Matched BF16 control summary_verification.json is missing")
    control_verification_record = read_json(control_verification_path)
    if not isinstance(control_verification_record, dict) or control_verification_record.get("passed") is not True:
        raise ValueError("Matched BF16 control summary verification did not pass")
    control_inputs = control_verification_record.get("input_sha256")
    if (not isinstance(control_inputs, dict) or
            control_inputs.get(str(frozen_path.resolve())) != sha256(frozen_path)):
        raise ValueError("Matched BF16 control verification does not hash the source summary")
    historical_path = Path(historical_summary_path) if historical_summary_path else None
    historical = read_json(historical_path) if historical_path else None
    if historical is not None:
        if not isinstance(historical, dict):
            raise ValueError("Historical frozen summary must contain a JSON object")
        if not isinstance(historical.get("seeds"), list) or not set(seeds).issubset(set(historical["seeds"])):
            raise ValueError("Historical summary does not contain all requested matched seeds")
        old_protocol = historical.get("protocol", {})
        if old_protocol.get("requested_training_episodes") != requested:
            raise ValueError("Historical and matched frozen training budgets differ")
        old_hard = next((row for row in historical.get("final_summary", [])
                         if row.get("split") == "hard" and row.get("policy") == "map"), None)
        if not isinstance(old_hard, dict) or old_hard.get("episodes_per_run") != hard_episodes:
            raise ValueError("Historical and matched frozen hard evaluation counts differ")

    run_rows = []
    matched_deltas: dict[str, dict[str, list[float]]] = {
        variant: {metric: [] for metric in METRICS} for variant in variants}
    did_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    historical_deltas: dict[str, dict[str, list[float]]] = {
        variant: {metric: [] for metric in METRICS} for variant in variants}
    for seed in seeds:
        frozen_seed = {variant: _load_frozen(frozen, seed, variant) for variant in variants}
        e2e_seed = {variant: _load_e2e(root, seed, variant, requested, hard_episodes)
                    for variant in variants}
        historical_seed = ({variant: _load_frozen(historical, seed, variant) for variant in variants}
                           if historical is not None else None)
        if historical_seed is not None:
            for variant, item in historical_seed.items():
                config = item["config"]
                if config.get("requested_episodes") != requested:
                    raise ValueError(f"Historical seed {seed}/{variant}: wrong requested training budget")
                preprocess = config.get("observation_signature", {}).get("preprocess", {})
                if "float16" not in str(preprocess.get("encoder_compute_precision", "")).lower():
                    raise ValueError(f"Historical seed {seed}/{variant}: expected the original FP16 frozen encoder")
        for variant in variants:
            f_metrics = frozen_seed[variant]["metrics"]
            e_metrics = e2e_seed[variant]["metrics"]
            _validate_match(frozen_seed[variant]["config"],
                            e2e_seed[variant]["training"]["config"],
                            e2e_seed[variant]["results"], seed, variant)
            for split in ("train", "hard"):
                if f_metrics[(split, variant)]["episodes"] != e_metrics[(split, variant)]["episodes"]:
                    raise ValueError(f"Seed {seed}/{variant}: frozen and e2e {split} episode counts differ")
                if historical_seed is not None and historical_seed[variant]["metrics"][(split, variant)]["episodes"] != f_metrics[(split, variant)]["episodes"]:
                    raise ValueError(f"Seed {seed}/{variant}: historical and matched frozen {split} episode counts differ")
            e_train = _row_metric(e_metrics[("train", variant)], "return_mean")
            e_hard = _row_metric(e_metrics[("hard", variant)], "return_mean")
            e_metrics[("hard", variant)]["generalization_gap_return"] = e_train - e_hard
            f_train = _row_metric(f_metrics[("train", variant)], "return_mean")
            f_hard = _row_metric(f_metrics[("hard", variant)], "return_mean")
            f_metrics[("hard", variant)]["generalization_gap_return"] = f_train - f_hard
            for metric in METRICS:
                f_value = _row_metric(f_metrics[("hard", variant)], metric)
                e_value = _row_metric(e_metrics[("hard", variant)], metric)
                if f_value is None or e_value is None:
                    continue
                matched_deltas[variant][metric].append(e_value - f_value)
                if historical_seed is not None:
                    old_metrics = historical_seed[variant]["metrics"]
                    old_train = _row_metric(old_metrics[("train", variant)], "return_mean")
                    old_hard = _row_metric(old_metrics[("hard", variant)], "return_mean")
                    old_metrics[("hard", variant)]["generalization_gap_return"] = old_train - old_hard
                    old_value = _row_metric(old_metrics[("hard", variant)], metric)
                    if old_value is not None:
                        historical_deltas[variant][metric].append(e_value - old_value)
        for metric in METRICS:
            emap = _row_metric(e2e_seed["map"]["metrics"][("hard", "map")], metric)
            eibe = _row_metric(e2e_seed["ibe_map"]["metrics"][("hard", "ibe_map")], metric)
            fmap = _row_metric(frozen_seed["map"]["metrics"][("hard", "map")], metric)
            fibe = _row_metric(frozen_seed["ibe_map"]["metrics"][("hard", "ibe_map")], metric)
            if None not in (emap, eibe, fmap, fibe):
                did_values[metric].append((eibe - emap) - (fibe - fmap))
        run_rows.append({"seed": seed, "frozen": {
            variant: {"metrics": _export_metrics(frozen_seed[variant]["metrics"], variant),
                      "results_path": frozen_seed[variant]["run"].get("results_path"),
                      "checkpoint_sha256": frozen_seed[variant]["run"].get("checkpoint_sha256", {}).get(variant)}
            for variant in variants}, "e2e": {
            variant: {"metrics": _export_metrics(e2e_seed[variant]["metrics"], variant),
                      "results_path": e2e_seed[variant]["results_path"],
                      "training_path": e2e_seed[variant]["training_path"],
                      "results_sha256": e2e_seed[variant]["results_sha256"],
                      "training_sha256": e2e_seed[variant]["training_sha256"],
                      "checkpoint_sha256": e2e_seed[variant]["checkpoint_sha256"],
                      "verification_path": e2e_seed[variant]["verification_path"],
                      "verification_sha256": e2e_seed[variant]["verification_sha256"],
                      "completed_episodes": e2e_seed[variant]["training"]["completed_episodes"],
                      "updates": e2e_seed[variant]["training"].get("updates"),
                      "config": e2e_seed[variant]["training"]["config"]}
            for variant in variants}, "historical": ({variant: {
                "metrics": _export_metrics(historical_seed[variant]["metrics"], variant),
                "results_path": historical_seed[variant]["run"].get("results_path"),
                "checkpoint_sha256": historical_seed[variant]["run"].get("checkpoint_sha256", {}).get(variant)}
                for variant in variants} if historical_seed is not None else None)})

    return {
        "schema_version": 1,
        "status": "complete",
        "seeds": list(seeds),
        "variants": list(variants),
        "training_budget_episodes": requested,
        "hard_episodes_per_run": hard_episodes,
        "frozen_summary_path": str(frozen_path.resolve()),
        "frozen_summary_sha256": sha256(frozen_path),
        "frozen_verification_sha256": sha256(control_verification_path),
        "frozen_verification_total": control_verification,
        "historical_summary_path": str(historical_path.resolve()) if historical_path else None,
        "historical_summary_sha256": sha256(historical_path) if historical_path else None,
        "e2e_root": str(root.resolve()),
        "run_count": len(seeds) * len(variants),
        "comparison": "same seed, same policy, e2e minus frozen; episodes and hard split counts must match",
        "statistics": "mean and sample SD (ddof=1); two-sided Student-t 95% CI across matched training seeds",
        "matched_deltas": {variant: _summary_block({k: v for k, v in metrics.items() if v})
                           for variant, metrics in matched_deltas.items()},
        "historical_paired_deltas_e2e_minus_fp16": ({variant: _summary_block({k: v for k, v in metrics.items() if v})
                                                      for variant, metrics in historical_deltas.items()}
                                                     if historical is not None else None),
        "difference_in_differences": {
            "definition": "(e2e IBE-MAP - e2e MAP) - (frozen IBE-MAP - frozen MAP)",
            "metrics": _summary_block({k: v for k, v in did_values.items() if v})},
        "final_runs": run_rows,
        "interpretation_limits": [
            "The frozen-to-e2e difference changes the ResNet50 training regime; it does not isolate SAM or GRU effects.",
            "The difference-in-differences estimates how the IBE-MAP versus MAP gap changes with encoder training regime under this protocol.",
            "Intervals describe variation across the matched training seeds on fixed scenes and fixed upstream data; they do not estimate new-scene uncertainty.",
            "With five seeds, comparisons are descriptive and do not receive a multiple-comparison correction."
        ]}


def write_report(summary: dict[str, Any], out_root: str | Path) -> tuple[Path, Path]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "comparison_summary.json"
    html_path = out_root / "comparison_report.html"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temporary.replace(summary_path)

    def table(title: str, groups: dict[str, Any]) -> str:
        rows = []
        for group, metrics in groups.items():
            for metric, stat in metrics.items():
                ci = stat.get("ci95_t")
                ci_text = "—" if ci is None else f"[{ci[0]:.5f}, {ci[1]:.5f}]"
                rows.append("<tr><td>{}</td><td>{}</td><td>{:.5f}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(group), html.escape(metric), stat["mean"],
                    "—" if stat.get("std") is None else f"{stat['std']:.5f}", ci_text))
        return (f"<h2>{html.escape(title)}</h2><table><thead><tr><th>比較</th><th>指標</th>"
                f"<th>均值</th><th>样本 SD</th><th>95% t 区间</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>")

    group_rows = []
    for regime, regimen_label in (("frozen", "冻结 ResNet50"),
                                  ("e2e", "端到端 ResNet50")):
        for variant, variant_label in (("map", "MAP"), ("ibe_map", "IBE-MAP（SAM+GRU）")):
            values = {metric: [] for metric in ("accuracy", "return_mean", "distance_m")}
            for run in summary.get("final_runs", []):
                hard = run.get(regime, {}).get(variant, {}).get("metrics", {}).get("hard", {})
                for metric in values:
                    values[metric].append(_finite_number(hard.get(metric), f"{regime}/{variant}/{metric}"))
            cells = []
            for metric, scale, digits in (("accuracy", 100, 2), ("return_mean", 1, 4), ("distance_m", 1, 2)):
                stats = seed_stats(values[metric])
                cells.append(f"{stats['mean'] * scale:.{digits}f} ± {stats['std'] * scale:.{digits}f}")
            group_rows.append(f"<tr><td>{html.escape(regimen_label)}</td><td>{html.escape(variant_label)}</td>"
                              + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    if summary.get("historical_summary_path"):
        for variant, variant_label in (("map", "MAP"), ("ibe_map", "IBE-MAP（SAM+GRU）")):
            values = {metric: [] for metric in ("accuracy", "return_mean", "distance_m")}
            for run in summary.get("final_runs", []):
                hard = run.get("historical", {}).get(variant, {}).get("metrics", {}).get("hard", {})
                for metric in values:
                    values[metric].append(_finite_number(hard.get(metric), f"historical/{variant}/{metric}"))
            cells = []
            for metric, scale, digits in (("accuracy", 100, 2), ("return_mean", 1, 4), ("distance_m", 1, 2)):
                stats = seed_stats(values[metric])
                cells.append(f"{stats['mean'] * scale:.{digits}f} ± {stats['std'] * scale:.{digits}f}")
            group_rows.append(f"<tr><td>历史冻结 ResNet50（FP16）</td><td>{html.escape(variant_label)}</td>"
                              + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    group_table = ("<table><thead><tr><th>編碼器</th><th>策略</th><th>準確率（%）均值 ± SD</th>"
                   "<th>平均回報均值 ± SD</th><th>移動距離（m）均值 ± SD</th></tr></thead><tbody>"
                   + "".join(group_rows) + "</tbody></table>")

    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结与端到端 ResNet50 比较</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 18px;color:#18212b}}table{{border-collapse:collapse;width:100%;margin:16px 0 30px}}th,td{{border:1px solid #ccd4dd;padding:8px;text-align:left}}th{{background:#edf2f7}}li{{margin:8px 0}}.meta{{color:#465363}}</style>
<h1>冻结与端到端 ResNet50 比较</h1><p class="meta">状态：完整；主对照为 BF16 冻结与 BF16 端到端训练；训练预算 {summary['training_budget_episodes']:,} episodes/策略；种子 {html.escape(', '.join(map(str, summary['seeds'])))}；每个 hard 评估 {summary['hard_episodes_per_run']:,} 回合。</p>
<p>主配对差值和 DiD 使用同 seed 的新 BF16 冻结对照。准确率单位为比例差；乘以 100 可读作百分点。DiD = (e2e IBE−MAP) − (matched frozen IBE−MAP)。历史 FP16 冻结结果仅作补充参考。</p>
<h2>各组 HARD 指标｜均值 ± 样本 SD</h2>{group_table}
<h2>准确率与平均回报</h2><p>误差线表示跨训练种子的 95% Student-t 区间。</p><img src="accuracy_return.png" alt="冻结与端到端 ResNet50 的准确率和平均回报比较" style="max-width:100%;height:auto">
{table('同策略端到端 − 冻结（HARD）', summary['matched_deltas'])}
{table('差异中的差异（HARD）', {'DiD': summary['difference_in_differences']['metrics']})}
{table('补充：端到端 BF16 − 历史冻结 FP16（同 seed；精度与可训练性均不同）', summary['historical_paired_deltas_e2e_minus_fp16']) if summary.get('historical_paired_deltas_e2e_minus_fp16') else ''}
<h2>逐种子记录</h2><p>机器可读逐 seed 指标、训练配置和文件哈希见 <a href="comparison_summary.json">comparison_summary.json</a>。</p>
<h2>解读限制</h2><ul>{''.join('<li>'+html.escape(item)+'</li>' for item in summary['interpretation_limits'])}</ul>
</html>"""
    html_path.write_text(page, encoding="utf-8")
    return summary_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="e2e experiment root")
    parser.add_argument("--frozen", required=True, type=Path, help="frozen multiseed summary.json")
    parser.add_argument("--historical", type=Path, help="optional historical FP16 frozen summary.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    summary = build_summary(args.frozen, args.out, tuple(args.seeds),
                            historical_summary_path=args.historical)
    summary_path, html_path = write_report(summary, args.out)
    print(f"Complete comparison: {summary_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
