"""Export a verified frozen-versus-end-to-end comparison for repository docs.

This module only packages an already audited complete report summary. It does
not train models, re-evaluate trajectories, or publish to a remote service.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from .report import DEFAULT_VARIANTS, read_json, seed_stats


PUBLISHED_METRICS = ("accuracy", "return_mean", "distance_m", "decisions",
                     "success_rate", "stop_rate", "budget_end_rate",
                     "generalization_gap_return")
VERIFICATION_KEYS = ("trajectories", "csv_rows", "unique_episode_keys")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hard_metric(row: dict[str, Any], metric: str) -> float:
    value = row.get("metrics", {}).get("hard", {}).get(metric)
    if value is None and metric == "generalization_gap_return":
        train = row.get("metrics", {}).get("train", {}).get("return_mean")
        hard = row.get("metrics", {}).get("hard", {}).get("return_mean")
        if train is not None and hard is not None:
            value = float(train) - float(hard)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Missing or invalid hard metric: {metric}")
    return float(value)


def _mean_summary(values: list[float]) -> dict[str, Any]:
    stats = seed_stats(values)
    return {key: stats[key] for key in ("mean", "std", "ci95_t", "n")}


def _compact_statistics(groups: dict[str, Any]) -> dict[str, Any]:
    return {group: {metric: {key: stat[key] for key in ("mean", "std", "ci95_t", "n")}
                    for metric, stat in metrics.items()}
            for group, metrics in groups.items()}


def _comparison_groups(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("final_runs")
    seeds = summary.get("seeds")
    variants = summary.get("variants")
    if not isinstance(runs, list) or not isinstance(seeds, list) or variants != list(DEFAULT_VARIANTS):
        raise ValueError("Comparison summary has an unexpected schema")
    expected = {(seed, variant) for seed in seeds for variant in DEFAULT_VARIANTS}
    observed = {(run.get("seed"), variant) for run in runs if isinstance(run, dict)
                for variant in run.get("e2e", {})}
    if len(runs) != len(expected) // len(DEFAULT_VARIANTS) or observed != expected:
        raise ValueError("Comparison summary does not contain the complete seed/policy grid")

    groups: dict[str, dict[str, Any]] = {}
    for label, regime, variant in (
            ("matched_frozen_bf16_map", "frozen", "map"),
            ("matched_frozen_bf16_ibe_map", "frozen", "ibe_map"),
            ("e2e_map", "e2e", "map"),
            ("e2e_ibe_map", "e2e", "ibe_map")):
        group_values = {metric: [] for metric in PUBLISHED_METRICS}
        for run in runs:
            block = run[regime][variant]
            for metric in PUBLISHED_METRICS:
                group_values[metric].append(_hard_metric(block, metric))
        groups[label] = {metric: _mean_summary(values) for metric, values in group_values.items()}
    if summary.get("historical_summary_path"):
        for variant, label in (("map", "historical_frozen_fp16_map"),
                               ("ibe_map", "historical_frozen_fp16_ibe_map")):
            group_values = {metric: [] for metric in PUBLISHED_METRICS}
            for run in runs:
                block = run.get("historical", {}).get(variant)
                if not isinstance(block, dict):
                    raise ValueError("Historical reference is incomplete")
                for metric in PUBLISHED_METRICS:
                    group_values[metric].append(_hard_metric(block, metric))
            groups[label] = {metric: _mean_summary(values) for metric, values in group_values.items()}
    return groups


def _write_plot(summary: dict[str, Any], destination: Path) -> None:
    """Render in a clean Python subprocess so torch and pyplot never share a process."""
    rows = summary["final_runs"]
    plot_groups = [("frozen", "map", "Matched BF16 MAP"),
                   ("frozen", "ibe_map", "Matched BF16 IBE-MAP"),
                   ("e2e", "map", "E2E BF16 MAP"),
                   ("e2e", "ibe_map", "E2E BF16 IBE-MAP")]
    if summary.get("historical_summary_path"):
        plot_groups.extend((
            ("historical", "map", "Historical FP16 MAP"),
            ("historical", "ibe_map", "Historical FP16 IBE-MAP")))
    plot_data: dict[str, Any] = {"labels": [label for _, _, label in plot_groups],
                                 "accuracy": [], "return_mean": []}
    for regime, variant, _ in plot_groups:
        for metric in ("accuracy", "return_mean"):
            values = [_hard_metric(run[regime][variant], metric) for run in rows]
            plot_data[metric].append(seed_stats(values))
    payload_path = destination.with_suffix(".plot-data.json")
    payload_path.write_text(json.dumps(plot_data, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    script = r'''import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open(sys.argv[1], encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for ax, metric, title, ylabel in (
    (axes[0], "accuracy", "Hard-split accuracy", "Accuracy"),
    (axes[1], "return_mean", "Hard-split mean return", "Mean discounted return")):
    rows = data[metric]
    labels = data["labels"]
    means = [row["mean"] for row in rows]
    cis = [row["ci95_t"] for row in rows]
    lower = [means[i] - cis[i][0] for i in range(len(rows))]
    upper = [cis[i][1] - means[i] for i in range(len(rows))]
    colors = ["#718096", "#4a5568", "#3182ce", "#805ad5", "#a0aec0", "#b794f4"][:len(rows)]
    ax.bar(range(len(rows)), means, color=colors, width=0.68)
    ax.errorbar(range(len(rows)), means, yerr=[lower, upper], fmt="none", ecolor="#1a202c", capsize=4, lw=1.3)
    ax.set_xticks(range(len(rows)), labels, rotation=18, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.22)
fig.suptitle("Frozen and end-to-end ResNet50 | mean with 95% t interval across seeds")
fig.savefig(sys.argv[2], dpi=180, bbox_inches="tight")
'''
    try:
        subprocess.run([sys.executable, "-c", script, str(payload_path), str(destination)],
                       check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise RuntimeError(f"Could not render publication plot: {detail}") from exc
    finally:
        payload_path.unlink(missing_ok=True)


def _run_metadata(summary: dict[str, Any], output_root: Path,
                   frozen_summary: dict[str, Any], historical_summary: dict[str, Any] | None = None
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_to_frozen = {run["seed"]: run for run in frozen_summary.get("final_runs", [])}
    seed_to_historical = ({run["seed"]: run for run in historical_summary.get("final_runs", [])}
                          if historical_summary else {})
    e2e_records, frozen_records, historical_records = [], [], []
    for run in summary["final_runs"]:
        seed = run["seed"]
        frozen_run = seed_to_frozen.get(seed)
        if not isinstance(frozen_run, dict):
            raise ValueError(f"Frozen source summary is missing seed {seed}")
        historical_run = seed_to_historical.get(seed)
        if historical_summary is not None and not isinstance(historical_run, dict):
            raise ValueError(f"Historical source summary is missing seed {seed}")
        for variant in DEFAULT_VARIANTS:
            source = run["e2e"][variant]
            base = output_root / f"seed_{seed}" / variant
            training_path = base / "train" / "training.json"
            checkpoint_path = base / "train" / "last.pt"
            verification_path = base / "evaluation" / "verification.json"
            if not all(path.is_file() for path in (training_path, checkpoint_path, verification_path)):
                raise ValueError(f"Incomplete audited E2E run files for seed {seed}/{variant}")
            training = read_json(training_path)
            config = training.get("config")
            if config != source.get("config"):
                raise ValueError(f"E2E config changed after summary for seed {seed}/{variant}")
            if source.get("training_sha256") and _sha256(training_path) != source["training_sha256"]:
                raise ValueError(f"E2E training record changed after summary for seed {seed}/{variant}")
            if _sha256(checkpoint_path) != source.get("checkpoint_sha256"):
                raise ValueError(f"E2E checkpoint changed after summary for seed {seed}/{variant}")
            verification = read_json(verification_path)
            if not isinstance(verification, dict) or verification.get("passed") is not True:
                raise ValueError(f"Independent trajectory verification failed for seed {seed}/{variant}")
            if source.get("verification_sha256") and _sha256(verification_path) != source["verification_sha256"]:
                raise ValueError(f"E2E verification record changed after summary for seed {seed}/{variant}")
            results_path = base / "evaluation" / "results.json"
            if not results_path.is_file():
                raise ValueError(f"Missing E2E evaluation results for seed {seed}/{variant}")
            results_digest = _sha256(results_path)
            if source.get("results_sha256") and results_digest != source["results_sha256"]:
                raise ValueError(f"E2E evaluation results changed after summary for seed {seed}/{variant}")
            input_hashes = verification.get("input_sha256")
            if input_hashes is not None and not isinstance(input_hashes, dict):
                raise ValueError(f"Invalid verification input hash map for seed {seed}/{variant}")
            if isinstance(input_hashes, dict):
                for artifact in (results_path, base / "evaluation" / "evaluation.csv",
                                 base / "evaluation" / "trajectories.jsonl.gz"):
                    resolved = artifact.resolve()
                    expected_digest = input_hashes.get(str(resolved))
                    if expected_digest is not None:
                        if not artifact.is_file() or _sha256(artifact) != expected_digest:
                            raise ValueError(f"Verified input changed after audit: {artifact}")
            counts = verification.get("counts", {})
            e2e_records.append({
                "seed": seed,
                "policy": variant,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "training_config_sha256": _canonical_sha256(config),
                "completed_episodes": training.get("completed_episodes"),
                "gradient_updates": training.get("updates"),
                "transitions": training.get("transitions"),
                "training_seconds": training.get("elapsed_seconds"),
                "evaluation_episodes": {split: source["metrics"][split]["episodes"] for split in ("train", "hard")},
                "verification_counts": {key: counts.get(key) for key in VERIFICATION_KEYS}})

            frozen_config = frozen_run["training"][variant]["config"]
            frozen_metrics = {row["split"]: row for row in frozen_run["metrics"] if row.get("policy") == variant}
            frozen_records.append({
                "seed": seed,
                "policy": variant,
                "checkpoint_sha256": frozen_run.get("checkpoint_sha256", {}).get(variant),
                "training_config_sha256": _canonical_sha256(frozen_config),
                "completed_episodes": frozen_run["training"][variant].get("completed_episodes"),
                "gradient_updates": frozen_run["training"][variant].get("updates"),
                "transitions": frozen_run["training"][variant].get("transitions"),
                "training_seconds": frozen_run["training"][variant].get("elapsed_seconds"),
                "evaluation_episodes": {split: frozen_metrics[split]["episodes"] for split in ("train", "hard")},
                "verification_counts": {key: frozen_run.get("verification_counts", {}).get(key) for key in VERIFICATION_KEYS}})
            if historical_run is not None:
                old_training = historical_run.get("training", {}).get(variant, {})
                old_config = old_training.get("config")
                old_metrics = {row["split"]: row for row in historical_run.get("metrics", [])
                               if row.get("policy") == variant}
                if not isinstance(old_config, dict) or not {"train", "hard"}.issubset(old_metrics):
                    raise ValueError(f"Historical source summary is incomplete for seed {seed}/{variant}")
                historical_records.append({
                    "seed": seed,
                    "policy": variant,
                    "checkpoint_sha256": historical_run.get("checkpoint_sha256", {}).get(variant),
                    "training_config_sha256": _canonical_sha256(old_config),
                    "completed_episodes": old_training.get("completed_episodes"),
                    "gradient_updates": old_training.get("updates"),
                    "transitions": old_training.get("transitions"),
                    "training_seconds": old_training.get("elapsed_seconds"),
                    "evaluation_episodes": {split: old_metrics[split]["episodes"] for split in ("train", "hard")},
                    "verification_counts": {key: historical_run.get("verification_counts", {}).get(key)
                                             for key in VERIFICATION_KEYS}})
    return e2e_records, frozen_records, historical_records


def _fmt_pm(stat: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    std = stat.get("std")
    if std is None:
        return f"{stat['mean'] * scale:.{digits}f}"
    return f"{stat['mean'] * scale:.{digits}f} ± {std * scale:.{digits}f}"


def _fmt_delta(stat: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    value = _fmt_pm(stat, scale, digits)
    interval = stat.get("ci95_t")
    if interval is None:
        return value
    return f"{value}; 95% t CI [{interval[0] * scale:.{digits}f}, {interval[1] * scale:.{digits}f}]"


def _zero_status(stat: dict[str, Any], positive: str, negative: str) -> str:
    interval = stat.get("ci95_t")
    if not isinstance(interval, list) or len(interval) != 2:
        return "区间不可用，方向不明确"
    low, high = interval
    if low > 0:
        return positive
    if high < 0:
        return negative
    return "区间跨 0，方向不明确"


def _auto_interpretation(metrics: dict[str, Any]) -> str:
    effects = (
        ("IBE-MAP 端到端 − 冻结", metrics["paired_deltas_e2e_minus_frozen"]["ibe_map"],
         {"accuracy": ("准确率较高", "准确率较低"),
          "return_mean": ("平均回报较高", "平均回报较低"),
          "distance_m": ("平均移动距离较长", "平均移动距离较短")}),
        ("DiD（端到端 IBE−MAP 减冻结 IBE−MAP）",
         metrics["difference_in_differences"]["metrics"],
         {"accuracy": ("IBE 相对 MAP 的准确率差扩大", "IBE 相对 MAP 的准确率差缩小"),
          "return_mean": ("IBE 相对 MAP 的回报差扩大", "IBE 相对 MAP 的回报差缩小"),
          "distance_m": ("IBE 相对 MAP 的移动距离差扩大", "IBE 相对 MAP 的移动距离差缩小")})
    )
    labels = {"accuracy": ("准确率", 100.0, 2, "个百分点"),
              "return_mean": ("平均折扣回报", 1.0, 4, ""),
              "distance_m": ("平均移动距离", 1.0, 2, " m")}
    lines = []
    for title, group, descriptions in effects:
        parts = []
        for metric in ("accuracy", "return_mean", "distance_m"):
            stat = group[metric]
            label, scale, digits, unit = labels[metric]
            ci = stat.get("ci95_t")
            if ci is None:
                interval_text = "95% t CI 不可用"
            else:
                interval_text = (f"95% t CI [{ci[0] * scale:.{digits}f}, "
                                 f"{ci[1] * scale:.{digits}f}]{unit}")
            status = _zero_status(stat, *descriptions[metric])
            parts.append(f"{label} {stat['mean'] * scale:+.{digits}f}{unit}（{interval_text}；{status}）")
        lines.append(f"- **{title}：**" + "；".join(parts) + "。")
    return "\n".join(lines)


def _write_readme(summary: dict[str, Any], metrics: dict[str, Any], destination: Path) -> None:
    core_metrics = (("accuracy", "准确率", 100.0, 2),
                    ("return_mean", "平均折扣回报", 1.0, 4),
                    ("distance_m", "平均移动距离（m）", 1.0, 2))
    labels = (("matched_frozen_bf16_map", "匹配冻结 ResNet50（BF16）· MAP"),
              ("matched_frozen_bf16_ibe_map", "匹配冻结 ResNet50（BF16）· IBE-MAP（SAM+GRU）"),
              ("e2e_map", "端到端 ResNet50（BF16）· MAP"),
              ("e2e_ibe_map", "端到端 ResNet50（BF16）· IBE-MAP（SAM+GRU）"))
    if metrics.get("historical_fp16_reference_runs") is not None:
        labels += (("historical_frozen_fp16_map", "历史冻结 ResNet50（FP16）· MAP"),
                   ("historical_frozen_fp16_ibe_map", "历史冻结 ResNet50（FP16）· IBE-MAP（SAM+GRU）"))
    group_rows = []
    for key, label in labels:
        cells = [_fmt_pm(metrics["group_summary"][key][metric], scale, digits)
                 for metric, _, scale, digits in core_metrics]
        group_rows.append(f"| {label} | " + " | ".join(cells) + " |")
    paired = ["| 策略 | 指标 | 端到端 − 冻结（均值 ± SD；95% t CI） |", "|---|---|---|"]
    for variant, label in (("map", "MAP"), ("ibe_map", "IBE-MAP / SAM+GRU")):
        for metric, label_metric, scale, digits in core_metrics:
            stat = metrics["paired_deltas_e2e_minus_frozen"][variant][metric]
            paired.append(f"| {label} | {label_metric} | {_fmt_delta(stat, scale, digits)} |")
    did = ["| 指标 | DiD（均值 ± SD；95% t CI） |", "|---|---|"]
    for metric, label, scale, digits in core_metrics:
        did.append(f"| {label} | {_fmt_delta(metrics['difference_in_differences']['metrics'][metric], scale, digits)} |")
    historical_paired = []
    supplemental = metrics.get("supplemental_paired_deltas_e2e_bf16_minus_historical_fp16")
    if supplemental:
        historical_paired = ["| 策略 | 指标 | E2E BF16 − 历史冻结 FP16（均值 ± SD；95% t CI） |", "|---|---|---|"]
        for variant, label in (("map", "MAP"), ("ibe_map", "IBE-MAP / SAM+GRU")):
            for metric, label_metric, scale, digits in core_metrics:
                stat = supplemental[variant][metric]
                historical_paired.append(f"| {label} | {label_metric} | {_fmt_delta(stat, scale, digits)} |")
    e2e_runs = metrics["e2e_runs"]
    actual_episodes = sum(row["completed_episodes"] for row in e2e_runs)
    total_seconds = sum(row["training_seconds"] for row in e2e_runs)
    verification = {key: sum(row["verification_counts"][key] or 0 for row in e2e_runs)
                    for key in VERIFICATION_KEYS}
    interpretation = _auto_interpretation(metrics)
    doc = f'''# Frozen and end-to-end ResNet50 results

## 中文结果

主对照为新生成的 BF16 冻结 ResNet50 MAP/IBE 与 BF16 端到端训练，使用 5 个匹配训练种子（{', '.join(map(str, metrics['seeds']))}）、相同的 300,000 回合请求预算、训练与验证场景、奖励和 hard split 起点。每个策略每个种子在 hard split 上评估 {metrics['hard_episodes_per_run']:,} 个共同起点。表中为五个训练种子的均值 ± 样本标准差（ddof=1）；主配对区间使用逐 seed 差值的双侧 Student-t 95% 区间。历史 FP16 冻结结果单列为补充参考。

| 模型组 | 准确率（%） | 平均折扣回报 | 平均移动距离（m） |
|---|---:|---:|---:|
{chr(10).join(group_rows)}

主同策略差值按“端到端 BF16 − 匹配冻结 BF16”计算；准确率差乘以 100 后以百分点表示。

{chr(10).join(paired)}

差异中的差异（DiD）定义为“端到端 BF16 IBE-MAP − 端到端 BF16 MAP”再减去“匹配冻结 BF16 IBE-MAP − 匹配冻结 BF16 MAP”。它描述 encoder 可训练性变化时 SAM+GRU 相对 MAP 的差值如何变化，不单独识别 SAM 或 GRU 的因果贡献。

{chr(10).join(did)}

{("E2E BF16 与历史冻结 FP16 的 seed 配对差值仅作为补充参考。此对照同时改变了数值精度和 encoder 可训练性，不能用于单独归因端到端训练效果。" + chr(10) + chr(10) + chr(10).join(historical_paired)) if historical_paired else ""}

### 自动解读

以下方向仅根据 seedwise 95% t 区间是否跨 0 判断；距离差为正表示移动距离更长，不代表性能提升。

{interpretation}

正式端到端实验包含 {len(e2e_runs)} 个完整训练任务，合计完成 {actual_episodes:,} 回合，累计训练时间 {total_seconds / 3600:.2f} 小时。轨迹独立核验计数合计：{verification['trajectories']:,} 条轨迹、{verification['csv_rows']:,} 行 CSV、{verification['unique_episode_keys']:,} 个唯一 episode 键。结果文件、checkpoint、训练配置哈希及逐 seed 计数见 [`metrics.json`](metrics.json)。图中的误差线为跨种子的 95% t 区间。

这些区间只描述固定场景和固定上游数据下训练种子的波动，不估计新场景或分类器重训的不确定性。种子数为 5，比较为描述性结果，未做多重比较校正。端到端与冻结之差改变了 ResNet50 的训练方式；不可将其解释成只改变 SAM 或 GRU 的实验。SAM 仍使用官方预训练模型，未在 UEVAVD 上微调。论文未说明其策略 backbone 是否冻结，也未说明 BatchNorm 的具体训练设置，因此本文仅报告当前实现下的对照结果。

## English brief

The primary comparison uses a newly trained BF16 frozen ResNet50 control against BF16 end-to-end ResNet50 training, with five matched seeds and the same requested 300,000-episode budget and hard-split evaluation starts. Values are mean ± sample SD. Primary paired differences and DiD use seed-level differences with two-sided 95% Student-t intervals. Historical FP16 frozen results are supplementary only because both precision and encoder trainability differ from the BF16 end-to-end runs.

SAM remains pretrained and is not fine-tuned on UEVAVD. The paper does not specify whether its policy backbone was frozen or how BatchNorm was handled, so this report describes the present implementation rather than attributing those choices to the paper.

![Hard-split accuracy and return, mean with 95% t intervals](accuracy_return.png)

## Result files

- `metrics.json` contains compact group summaries, primary paired deltas, DiD, supplemental historical comparisons, and hashes and counts for the 10 end-to-end runs and frozen references.
- `accuracy_return.png` compares hard-split accuracy and mean return across the reported groups.
'''
    destination.write_text(doc, encoding="utf-8")


def export_results(summary: dict[str, Any], output_root: str | Path,
                   destination: str | Path) -> dict[str, Path]:
    """Export audited complete results to a compact docs directory.

    ``summary`` must be the completed output of ``e2e_map.report.build_summary``.
    Local source paths are used only to read/check artifacts and are omitted from
    the published JSON and Markdown.
    """
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise ValueError("Only a complete audited comparison can be exported")
    output_root, destination = Path(output_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if summary.get("e2e_root") and Path(summary["e2e_root"]).resolve() != output_root.resolve():
        raise ValueError("Output root does not match the audited comparison summary")
    frozen_path = summary.get("frozen_summary_path")
    if not frozen_path or _sha256(frozen_path) != summary.get("frozen_summary_sha256"):
        raise ValueError("Frozen source summary is missing or changed after auditing")
    frozen_source = read_json(frozen_path)
    frozen_verification_path = Path(frozen_path).parent / "summary_verification.json"
    if (not frozen_verification_path.is_file() or
            _sha256(frozen_verification_path) != summary.get("frozen_verification_sha256")):
        raise ValueError("Matched BF16 control verification source is missing or changed after auditing")
    frozen_verification = read_json(frozen_verification_path)
    if not isinstance(frozen_verification, dict) or frozen_verification.get("passed") is not True:
        raise ValueError("Matched BF16 control summary verification did not pass")
    frozen_inputs = frozen_verification.get("input_sha256")
    if (not isinstance(frozen_inputs, dict) or
            frozen_inputs.get(str(Path(frozen_path).resolve())) != _sha256(frozen_path)):
        raise ValueError("Matched BF16 control verification no longer covers its source summary")
    historical_path = summary.get("historical_summary_path")
    historical_source = None
    if historical_path:
        if _sha256(historical_path) != summary.get("historical_summary_sha256"):
            raise ValueError("Historical FP16 source summary is missing or changed after auditing")
        historical_source = read_json(historical_path)

    groups = _comparison_groups(summary)
    e2e_runs, frozen_runs, historical_runs = _run_metadata(
        summary, output_root, frozen_source, historical_source)
    if len(e2e_runs) != 10 or len(frozen_runs) != 10:
        raise ValueError("Publication requires exactly 10 E2E runs and 10 matched frozen controls")
    if historical_source is not None and len(historical_runs) != 10:
        raise ValueError("Historical FP16 reference is missing one or more seed/policy records")
    manifest_path = output_root / "run_manifest.json"
    training_totals = {
        "run_count": len(e2e_runs),
        "requested_episodes_per_run": summary.get("training_budget_episodes"),
        "actual_completed_episodes": sum(row["completed_episodes"] for row in e2e_runs),
        "gradient_updates": sum(row["gradient_updates"] for row in e2e_runs),
        "transitions": sum(row["transitions"] for row in e2e_runs),
        "summed_training_seconds": sum(row["training_seconds"] for row in e2e_runs)}
    verification_totals = {key: sum(row["verification_counts"][key] or 0 for row in e2e_runs)
                           for key in VERIFICATION_KEYS}
    limitations = list(summary.get("interpretation_limits", []))
    for note in (
            "SAM uses official pretrained weights and is not fine-tuned on UEVAVD.",
            "The paper does not specify whether its policy backbone was frozen or how BatchNorm was handled."):
        if note not in limitations:
            limitations.append(note)
    paired = summary.get("matched_deltas")
    did = summary.get("difference_in_differences", {}).get("metrics")
    for variant in DEFAULT_VARIANTS:
        if not isinstance(paired, dict) or variant not in paired:
            raise ValueError(f"Missing paired summary for {variant}")
    if not isinstance(did, dict):
        raise ValueError("Missing difference-in-differences summary")

    historical_deltas = summary.get("historical_paired_deltas_e2e_minus_fp16")
    if historical_source is not None and not isinstance(historical_deltas, dict):
        raise ValueError("Historical paired deltas are missing from the audited comparison")
    published = {
        "project": "UEVAVD: Frozen and End-to-End ResNet50 Comparison",
        "schema_version": 1,
        "status": "complete",
        "seeds": summary["seeds"],
        "training_budget_episodes": summary["training_budget_episodes"],
        "hard_episodes_per_run": summary["hard_episodes_per_run"],
        "statistical_method": "mean and sample SD (ddof=1); two-sided Student-t 95% CI across five training seeds; paired intervals use seedwise differences",
        "group_summary": groups,
        "paired_deltas_e2e_minus_frozen": _compact_statistics(paired),
        "supplemental_paired_deltas_e2e_bf16_minus_historical_fp16": (
            _compact_statistics(historical_deltas) if historical_deltas is not None else None),
        "difference_in_differences": {
            "definition": summary["difference_in_differences"]["definition"],
            "metrics": _compact_statistics({"DiD": did})["DiD"]},
        "training_totals_e2e": training_totals,
        "verification_totals_e2e": verification_totals,
        "verification_totals_matched_frozen": summary.get("frozen_verification_total"),
        "source_hashes": {
            "frozen_summary_sha256": summary["frozen_summary_sha256"],
            "frozen_summary_verification_sha256": summary["frozen_verification_sha256"],
            "historical_fp16_summary_sha256": summary.get("historical_summary_sha256"),
            "e2e_run_manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None},
        "e2e_runs": e2e_runs,
        "frozen_control_runs": frozen_runs,
        "historical_fp16_reference_runs": historical_runs if historical_source is not None else None,
        "interpretation_limits": limitations}
    metrics_path = destination / "metrics.json"
    temp_path = metrics_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(published, indent=2, ensure_ascii=True,
                                    allow_nan=False), encoding="utf-8")
    temp_path.replace(metrics_path)
    plot_path = destination / "accuracy_return.png"
    _write_plot(summary, plot_path)
    readme_path = destination / "README.md"
    _write_readme(summary, published, readme_path)
    return {"readme": readme_path, "metrics": metrics_path, "plot": plot_path}
