"""Paired scene-level ablation analysis for IBE-MAP evaluation exports."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 1729
COMPARISONS = (
    ("memo_map", "map", "Memo-MAP − MAP", "GRU effect without SAM"),
    ("sam_map", "map", "SAM-MAP − MAP", "SAM effect without GRU"),
    ("ibe_map", "memo_map", "IBE-MAP − Memo-MAP", "SAM effect with GRU"),
    ("ibe_map", "sam_map", "IBE-MAP − SAM-MAP", "GRU effect with SAM"),
    ("ibe_map", "map", "IBE-MAP − MAP", "Combined effect"),
)
METRICS = ("accuracy", "return_mean", "distance_m")
START_FIELDS = ("start_x", "start_y", "start_z")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value).strip().casefold()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean in evaluation CSV: {value!r}")


def _float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in evaluation CSV: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field} in evaluation CSV: {value!r}")
    return result


def _start_key(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(_float(row[name], name) for name in START_FIELDS)  # type: ignore[return-value]


def _scene_rows(csv_path: Path) -> dict[str, dict[str, dict[tuple[float, float, float], dict[str, float]]]]:
    """Return policy -> scene -> start key -> values for hard native episodes."""
    policies: dict[str, dict[str, dict[tuple[float, float, float], dict[str, float]]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "mode", "policy", "scene", *START_FIELDS, "final_correct", "return", "distance_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Evaluation CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if (row.get("split", "").strip().casefold() != "hard"
                    or row.get("mode", "").strip().casefold() != "native"):
                continue
            policy = row["policy"].strip()
            scene = str(row["scene"]).strip()
            if not policy or not scene:
                raise ValueError("Hard native rows need non-empty policy and scene")
            start = _start_key(row)
            starts = policies.setdefault(policy, {}).setdefault(scene, {})
            if start in starts:
                raise ValueError(f"Duplicate hard native episode for policy={policy}, scene={scene}, start={start}")
            starts[start] = {
                "accuracy": float(_parse_bool(row["final_correct"])),
                "return_mean": _float(row["return"], "return"),
                "distance_m": _float(row["distance_m"], "distance_m"),
            }
    return policies


def _aggregate(policies: dict[str, dict[str, dict[tuple[float, float, float], dict[str, float]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy, scenes in policies.items():
        result[policy] = {}
        for scene, starts in scenes.items():
            result[policy][scene] = {
                "episodes": len(starts),
                **{metric: float(np.mean([values[metric] for values in starts.values()])) for metric in METRICS},
            }
    return result


def _compare(
    policies: dict[str, dict[str, dict[tuple[float, float, float], dict[str, float]]]],
    left: str,
    right: str,
) -> dict[str, Any]:
    left_scenes, right_scenes = policies[left], policies[right]
    scenes = sorted(set(left_scenes) & set(right_scenes), key=str)
    mismatches = []
    if set(left_scenes) != set(right_scenes):
        mismatches.append({"kind": "scene_set", "left_only": sorted(set(left_scenes)-set(right_scenes)),
                           "right_only": sorted(set(right_scenes)-set(left_scenes))})
    for scene in scenes:
        lk, rk = set(left_scenes[scene]), set(right_scenes[scene])
        if lk != rk:
            mismatches.append({"kind": "start_set", "scene": scene, "left_only": len(lk-rk), "right_only": len(rk-lk)})
    entry: dict[str, Any] = {"left_policy": left, "right_policy": right,
                             "n_scenes": len(scenes), "protocol_valid": not mismatches,
                             "validation_errors": mismatches, "metrics": {}}
    if mismatches or not scenes:
        return entry

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in METRICS:
        scene_diffs = np.asarray([
            np.mean([left_scenes[s][k][metric] - right_scenes[s][k][metric]
                     for k in left_scenes[s]])
            for s in scenes
        ], dtype=float)
        # Scene means are equally weighted, equivalent to the pooled episode mean
        # only when every scene has the same number of starting viewpoints.
        draws = rng.integers(0, len(scene_diffs), size=(BOOTSTRAP_DRAWS, len(scene_diffs)))
        boot_means = scene_diffs[draws].mean(axis=1)
        point = float(scene_diffs.mean())
        unit = "percentage_points" if metric == "accuracy" else ("return" if metric == "return_mean" else "meters")
        scale = 100.0 if metric == "accuracy" else 1.0
        entry["metrics"][metric] = {
            "difference": point * scale,
            "ci95": [float(v) * scale for v in np.quantile(boot_means, [0.025, 0.975])],
            "unit": unit,
        }
    return entry


def analyze(results_path: str | Path, out: str | Path) -> dict[str, Any]:
    """Write `ablation.json` and return the data plus its output path.

    `results_path` is required as the experiment's provenance record. Episode-level
    paired analysis is read from `evaluation.csv` beside it.
    """
    results_path, out = Path(results_path), Path(out)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(results, dict):
        raise ValueError(f"Results JSON must contain an object: {results_path}")
    csv_path = results_path.with_name("evaluation.csv")
    policies = _scene_rows(csv_path)
    aggregates = _aggregate(policies)

    comparisons = []
    for left, right, label, interpretation in COMPARISONS:
        if left in policies and right in policies:
            comparison = _compare(policies, left, right)
            comparison["label"] = label
            comparison["interpretation"] = interpretation
            comparisons.append(comparison)

    if out.suffix.casefold() == ".json":
        output_path = out
    else:
        output_path = out / "ablation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "results_path": str(results_path.resolve()),
        "evaluation_csv": str(csv_path.resolve()),
        "split": "hard",
        "mode": "native",
        "weighting": "equal weight per scene; paired by identical start_x/start_y/start_z",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "ci": "percentile 95%"},
        "scene_metrics": aggregates,
        "comparisons": comparisons,
        "limitations": [
            "A single training seed's context bootstrap interval reflects variation across evaluated scenes, not training-seed uncertainty or proof of statistical significance across seeds.",
            "Comparisons are observational ablations on the fixed protocol; they must not be used to tune on the hard test split.",
        ],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": output_path, "data": report}


__all__ = ["analyze"]
