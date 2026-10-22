"""Offline report for the five-seed UEVAVD policy comparison.

The renderer only consumes the aggregate JSON and per-seed artifact paths. It
does not import training code, torch, or the evaluation environment.
"""
from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_ORDER = ("stationary", "sweep", "random", "map", "memo_map", "sam_map", "ibe_map")
LEARNED = ("map", "memo_map", "sam_map", "ibe_map")
POLICY_LABELS = {
    "stationary": "Stationary / 固定", "sweep": "Sweep / 掃描", "random": "Random / 隨機",
    "map": "MAP", "memo_map": "Memo-MAP", "sam_map": "SAM-MAP", "ibe_map": "IBE-MAP",
}
COLORS = {
    "stationary": "#778899", "sweep": "#4778a8", "random": "#dd9b32",
    "map": "#59a14f", "memo_map": "#af7aa1", "sam_map": "#e15759", "ibe_map": "#2f7f8f",
}
METRICS = {
    "accuracy": ("準確率", "%", 100.0),
    "return_mean": ("平均回報", "回報", 1.0),
    "distance_m": ("移動距離", "公尺", 1.0),
}
SPLIT_LABELS = {"train": "TRAIN", "hard": "HARD"}
T95_DF4 = 2.7764451051977987
T95_BY_N = {
    2: 12.706204736, 3: 4.30265273, 4: 3.182446305, 5: T95_DF4,
    6: 2.570581836, 7: 2.446911851, 8: 2.364624252,
    9: 2.306004135, 10: 2.262157163,
}


def _t_critical(n: int) -> float:
    if n <= 1:
        raise ValueError("A Student-t interval needs at least two independent samples")
    if n not in T95_BY_N:
        raise ValueError(f"Student-t critical value is not configured for n={n}; supported sample sizes are 2–10")
    return T95_BY_N[n]


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _label(policy: Any) -> str:
    p = str(policy or "")
    return POLICY_LABELS.get(p, p.replace("_", " "))


def _fmt(value: Any, digits: int = 3) -> str:
    x = _finite(value)
    return f"{x:.{digits}f}" if x is not None else "—"


def _metric(row: dict[str, Any], key: str) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    raw = metrics.get(key, row.get(key))
    if isinstance(raw, dict):
        return raw
    return {"mean": raw}


def _mean(metric: dict[str, Any]) -> float | None:
    v = _finite(metric.get("mean"))
    if v is not None:
        return v
    vals = [_finite(x) for x in metric.get("values", [])] if isinstance(metric.get("values"), list) else []
    vals = [x for x in vals if x is not None]
    return sum(vals) / len(vals) if vals else None


def _std(metric: dict[str, Any]) -> float | None:
    v = _finite(metric.get("std"))
    if v is not None:
        return v
    vals = [_finite(x) for x in metric.get("values", [])] if isinstance(metric.get("values"), list) else []
    vals = [x for x in vals if x is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))


def _ci(metric: dict[str, Any]) -> tuple[float, float] | None:
    ci = metric.get("ci95_t") or metric.get("ci95")
    if isinstance(ci, (tuple, list)) and len(ci) == 2:
        lo, hi = _finite(ci[0]), _finite(ci[1])
        if lo is not None and hi is not None:
            return lo, hi
    mean, sd = _mean(metric), _std(metric)
    n = int(_finite(metric.get("n")) or 0)
    if mean is not None and sd is not None and n > 1:
        # Summary protocol is five seeds (df=4); retain the correct df when a
        # smaller fixture or partial run is rendered.
        critical = _t_critical(n)
        half = critical * sd / math.sqrt(n)
        return mean - half, mean + half
    return None


def _hard_row(summary: dict[str, Any], policy: str) -> dict[str, Any] | None:
    for row in summary.get("final_summary", []):
        if isinstance(row, dict) and row.get("split") == "hard" and row.get("policy") == policy:
            return row
    return None


def _numfmt(v: Any, digits: int = 3, scale: float = 1.0) -> str:
    x = _finite(v)
    return f"{x * scale:.{digits}f}" if x is not None else "—"


def _mean_sd_cell(metric: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    mean, sd = _mean(metric), _std(metric)
    if mean is None:
        return "—"
    if int(_finite(metric.get("n")) or 0) <= 1:
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {sd * scale:.{digits}f}" if sd is not None else f"{mean * scale:.{digits}f}"


def _ci_cell(metric: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    ci = _ci(metric)
    if ci is None:
        return "不適用" if int(_finite(metric.get("n")) or 0) <= 1 else "—"
    return f"[{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}]"


def _hard_table(summary: dict[str, Any]) -> str:
    rows = []
    for policy in POLICY_ORDER:
        row = _hard_row(summary, policy)
        if row is None:
            continue
        randomness = row.get("randomness", "training_seed")
        randomness_label = {"deterministic": "確定性參考", "evaluation_seed": "evaluation seeds", "training_seed": "training seeds"}.get(str(randomness), randomness)
        metrics = {key: _metric(row, key) for key in (*METRICS, "generalization_gap_return")}
        cells = [f"<td>{_esc(_label(policy))}</td>", f"<td>{_esc(randomness_label)}</td>"]
        for key in METRICS:
            scale = METRICS[key][2]
            digits = 2 if key == "accuracy" else 3
            cells.append(f"<td>{_mean_sd_cell(metrics[key], scale, digits)}</td>")
        cells.append(f"<td>{_ci_cell(metrics['return_mean'], 1, 3)}</td>")
        cells.append(f"<td>{_mean_sd_cell(metrics['generalization_gap_return'], 1, 3)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<div class='scroll'><table><thead><tr><th>策略</th><th>種子來源</th><th>準確率 (%) 均值 ± SD</th><th>回報均值 ± SD</th><th>距離 (m) 均值 ± SD</th><th>回報 95% t CI</th><th>GenGap (TRAIN−HARD)</th></tr></thead><tbody>" + ("".join(rows) or "<tr><td colspan='7'>摘要尚無 HARD 結果。</td></tr>") + "</tbody></table></div>"


def _paired_table(summary: dict[str, Any]) -> str:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary.get("paired_comparisons", []):
        if isinstance(row, dict) and row.get("split") == "hard":
            groups[(str(row.get("left_policy")), str(row.get("right_policy")))] = row
    rows = []
    for i, left in enumerate(LEARNED):
        for right in LEARNED[i + 1:]:
            row = groups.get((left, right)) or groups.get((right, left))
            if not row:
                continue
            flipped = row.get("left_policy") != left
            cells = [f"<td>{_esc(_label(left))} − {_esc(_label(right))}</td>"]
            for key in (*METRICS.keys(), "generalization_gap_return"):
                m = _metric(row, key)
                sign = -1 if flipped else 1
                value = _mean(m)
                ci = _ci(m)
                scale = METRICS.get(key, ("", "", 1.0))[2]
                digits = 2 if key == "accuracy" else 3
                if value is None:
                    cell = "—"
                else:
                    cell = f"{sign * value * scale:+.{digits}f}"
                    if ci:
                        bounds = sorted((sign * ci[0] * scale, sign * ci[1] * scale))
                        cell += f" [{bounds[0]:+.{digits}f}, {bounds[1]:+.{digits}f}]"
                cells.append(f"<td>{cell}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<div class='scroll'><table><thead><tr><th>配對差值（左 − 右）</th><th>Δ 準確率</th><th>Δ 回報</th><th>Δ 距離</th><th>Δ GenGap</th></tr></thead><tbody>" + ("".join(rows) or "<tr><td colspan='5'>尚無四個學習策略的 HARD 同 seed 配對比較。</td></tr>") + "</tbody></table></div>"


def _curve_value(record: dict[str, Any], key: str) -> float | None:
    return _mean(_metric(record, key))


def _curve_points(record: dict[str, Any], metric: str) -> list[tuple[float, float, float]]:
    """Return (requested budget, actual episodes, value) for checkpoint rows."""
    points: list[tuple[float, float, float]] = []
    history = record.get("history")
    records = history if isinstance(history, list) and history else [record]
    for item in records:
        if not isinstance(item, dict):
            continue
        actual = _finite(item.get("actual_episodes", item.get("episodes", item.get("checkpoint_episodes"))))
        requested = _finite(item.get("requested_episodes", item.get("checkpoint_episodes", item.get("episodes"))))
        if requested is None:
            requested = _finite(record.get("requested_episodes", record.get("checkpoint_episodes")))
        if requested is None:
            requested = actual
        if actual is None:
            actual = requested
        value = _curve_value(item, metric)
        if requested is not None and actual is not None and value is not None:
            points.append((requested, actual, value))
    return points


def _curve_chart(curves: list[dict[str, Any]], split: str, metric: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.5), layout="constrained")
    found = False
    for policy in LEARNED:
        records = [r for r in curves if str(r.get("policy")) == policy and str(r.get("split")) == split]
        if not records:
            continue
        # Join seeds at the requested checkpoint budget. Parallel environments
        # can finish a few extra episodes, so actual_episodes is not a key.
        series_by_seed: dict[str, dict[float, tuple[float, float]]] = {}
        for rec in records:
            seed = str(rec.get("seed", "?"))
            for requested, actual, value in _curve_points(rec, metric):
                series_by_seed.setdefault(seed, {})[requested] = (actual, value)
        if not series_by_seed:
            continue
        color = COLORS[policy]
        seed_maps = [series_by_seed[seed] for seed in sorted(series_by_seed)]
        for points in seed_maps:
            budgets = sorted(points)
            actual_xs = [points[budget][0] for budget in budgets]
            ys = [points[budget][1] * METRICS[metric][2] for budget in budgets]
            # Keep individual paths in their observed episode coordinates.
            ax.plot(actual_xs, ys, color="#7f8892", alpha=.23, linewidth=.9, zorder=1)
        common_budgets = sorted(set.intersection(*(set(m) for m in seed_maps))) if seed_maps else []
        if common_budgets:
            matrix = np.array([[m[b][1] for b in common_budgets] for m in seed_maps], dtype=float)
            mean_actual = np.array([np.mean([m[b][0] for m in seed_maps]) for b in common_budgets])
            means = np.mean(matrix, axis=0)
            if len(seed_maps) > 1:
                sd = np.std(matrix, axis=0, ddof=1)
                half = _t_critical(len(seed_maps)) * sd / math.sqrt(len(seed_maps))
            else:
                half = np.zeros_like(means)
            scale = METRICS[metric][2]
            xs = mean_actual
            means, half = means * scale, half * scale
            ax.fill_between(xs, means - half, means + half, color=color, alpha=.16, linewidth=0, zorder=2)
            ax.plot(xs, means, color=color, linewidth=2.3, label=_label(policy), zorder=3)
            found = True
    metric_title = {"accuracy": "Accuracy (%)", "return_mean": "Mean return", "distance_m": "Travel distance (m)"}[metric]
    ax.set_title(f"{SPLIT_LABELS.get(split, split.upper())} | {metric_title}")
    ax.set_xlabel("Actual episodes (mean across matched requested checkpoints)")
    ax.set_ylabel(metric_title)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)
    ax.ticklabel_format(style="plain", axis="x")
    if found:
        ax.legend(frameon=False, ncol=2, fontsize=8)
    else:
        ax.text(.5, .5, "摘要中尚無可繪製的訓練曲線", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def _gap_chart(curves: list[dict[str, Any]], out: Path) -> None:
    # Pair TRAIN and HARD return at the same seed and checkpoint.
    keyed: dict[tuple[str, str, float], dict[str, tuple[float, float]]] = {}
    for rec in curves:
        p, split = str(rec.get("policy", "")), str(rec.get("split", ""))
        if p not in LEARNED or split not in ("train", "hard"):
            continue
        seed = str(rec.get("seed", "?"))
        for requested, actual, value in _curve_points(rec, "return_mean"):
            keyed.setdefault((p, seed, requested), {})[split] = (actual, value)
    data = [{"seed": seed, "requested_episodes": requested,
             "actual_episodes": (s["train"][0] + s["hard"][0]) / 2,
             "gap": s["train"][1] - s["hard"][1], "policy": p}
            for (p, seed, requested), s in keyed.items() if "train" in s and "hard" in s]
    _generic_curve_chart(data, "gap", "GenGap: TRAIN - HARD mean return", "Return difference", out)


def _generic_curve_chart(data: list[dict[str, Any]], value_key: str, title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.5), layout="constrained")
    found = False
    for policy in LEARNED:
        rows = [r for r in data if r["policy"] == policy]
        by_seed: dict[str, dict[float, tuple[float, float]]] = {}
        for r in rows:
            by_seed.setdefault(r["seed"], {})[r["requested_episodes"]] = (r["actual_episodes"], r[value_key])
        maps = list(by_seed.values())
        color = COLORS[policy]
        for m in maps:
            budgets = sorted(m)
            ax.plot([m[b][0] for b in budgets], [m[b][1] for b in budgets],
                    color="#7f8892", alpha=.23, linewidth=.9, zorder=1)
        common_budgets = sorted(set.intersection(*(set(m) for m in maps))) if maps else []
        if common_budgets:
            matrix = np.array([[m[b][1] for b in common_budgets] for m in maps])
            xs = np.array([np.mean([m[b][0] for m in maps]) for b in common_budgets])
            mean = matrix.mean(axis=0)
            if len(maps) > 1:
                sd = matrix.std(axis=0, ddof=1)
                half = _t_critical(len(maps)) * sd / math.sqrt(len(maps))
            else:
                half = np.zeros_like(mean)
            ax.fill_between(xs, mean - half, mean + half, color=color, alpha=.16, linewidth=0, zorder=2)
            ax.plot(xs, mean, color=color, linewidth=2.3, label=_label(policy), zorder=3)
            found = True
    ax.set_title(title)
    ax.set_xlabel("Actual episodes (mean across matched requested checkpoints)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.22); ax.set_axisbelow(True); ax.ticklabel_format(style="plain", axis="x")
    if found:
        ax.legend(frameon=False, ncol=2, fontsize=8)
    else:
        ax.text(.5, .5, "TRAIN/HARD 成對曲線尚未提供", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def _relative_link(target: Any, html_path: Path, label: str) -> str:
    if not target:
        return "—"
    p = Path(str(target))
    if not p.is_absolute():
        p = (html_path.parent / p).resolve()
    try:
        rel = os.path.relpath(p, html_path.parent.resolve()).replace(os.sep, "/")
    except ValueError:
        rel = str(p).replace(os.sep, "/")
    href = quote(rel, safe="/._-#")
    return f"<a href=\"{_esc(href)}\">{_esc(label)}</a>"


def _seed_section(summary: dict[str, Any], html_path: Path) -> str:
    rows = []
    runs = summary.get("final_runs", [])
    if not isinstance(runs, list):
        runs = []
    for run in sorted((r for r in runs if isinstance(r, dict)), key=lambda r: (int(_finite(r.get("seed")) or 0))):
        seed = run.get("seed", "?")
        results = _relative_link(run.get("results_path"), html_path, "結果 JSON")
        verify = _relative_link(run.get("verification_path"), html_path, "驗證報告")
        examples = _relative_link(run.get("examples_path"), html_path, "軌跡與示例")
        rows.append(f"<tr><td>{_esc(seed)}</td><td>{results}</td><td>{verify}</td><td>{examples}</td></tr>")
    return "<div class='scroll'><table><thead><tr><th>Seed</th><th>評估結果</th><th>驗證狀態</th><th>軌跡 / 示例</th></tr></thead><tbody>" + ("".join(rows) or "<tr><td colspan='4'>尚無逐 seed artifacts。</td></tr>") + "</tbody></table></div>"


def _verification(summary: dict[str, Any]) -> str:
    total = summary.get("verification_total")
    if not isinstance(total, dict):
        return "尚未提供驗證計數。"
    bits = []
    for key, value in total.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        bits.append(f"<span><b>{_esc(str(key).replace('_', ' '))}</b> {_esc(value)}</span>")
    return "　·　".join(bits)


def _training_facts(summary: dict[str, Any]) -> str:
    facts = summary.get("training") if isinstance(summary.get("training"), dict) else {}
    protocol = summary.get("protocol") if isinstance(summary.get("protocol"), dict) else {}
    combined = dict(facts)
    for k, v in protocol.items():
        combined.setdefault(k, v)
    for k in ("requested_episodes", "actual_episodes", "budget_episodes", "episodes_per_policy", "checkpoint_episodes", "seeds", "policies", "encoder", "sam", "validation"):
        if k in summary:
            combined.setdefault(k, summary[k])
    if not combined:
        # Derive budget/progress from actual curve records instead of implying
        # the requested budget was completed.
        curves = summary.get("curves", [])
        actual = [_finite(r.get("actual_episodes")) for r in curves if isinstance(r, dict)] if isinstance(curves, list) else []
        actual = [x for x in actual if x is not None]
        combined = {"completed checkpoint maximum": max(actual) if actual else "未提供"}
    return "".join(f"<div><small>{_esc(str(k).replace('_', ' ').title())}</small><strong>{_esc(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))}</strong></div>" for k, v in combined.items())


def _checkpoint_facts(summary: dict[str, Any], curves: list[dict[str, Any]]) -> tuple[list[float], float | None]:
    training = summary.get("training") if isinstance(summary.get("training"), dict) else {}
    protocol = summary.get("protocol") if isinstance(summary.get("protocol"), dict) else {}
    raw = protocol.get("checkpoint_budgets", training.get("checkpoints", summary.get("checkpoints")))
    checkpoints = [_finite(v) for v in raw] if isinstance(raw, list) else []
    checkpoints = sorted(set(v for v in checkpoints if v is not None))
    if not checkpoints:
        checkpoints = sorted(set(v for r in curves if isinstance(r, dict)
                                 for v in [_finite(r.get("requested_episodes", r.get("checkpoint_episodes")))]
                                 if v is not None))
    budget = _finite(protocol.get("requested_training_episodes"))
    if budget is None:
        budget = _finite(training.get("episodes_per_policy", training.get("budget_episodes", training.get("requested_episodes"))))
    if budget is None:
        budget = _finite(summary.get("episodes_per_policy", summary.get("budget_episodes", summary.get("requested_episodes"))))
    if budget is None and checkpoints:
        budget = max(checkpoints)
    return checkpoints, budget


def render_report(summary_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Create an offline multi-seed HTML report and PNG figures.

    ``summary_path`` is the runner's summary.json. ``out_dir`` is created if
    needed. Returns paths keyed by ``html`` and generated chart names.
    """
    summary_path, out_dir = Path(summary_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain a JSON object")
    curves = summary.get("curves", [])
    if not isinstance(curves, list):
        curves = []
    checkpoints, budget = _checkpoint_facts(summary, curves)
    chart_paths: dict[str, Path] = {}
    for split in ("train", "hard"):
        for metric in METRICS:
            name = f"curve_{split}_{metric}.png"
            path = out_dir / name
            _curve_chart([r for r in curves if isinstance(r, dict)], split, metric, path)
            chart_paths[name] = path
    gap_path = out_dir / "curve_gengap.png"
    _gap_chart([r for r in curves if isinstance(r, dict)], gap_path)
    chart_paths[gap_path.name] = gap_path
    html_path = out_dir / "report.html"
    curve_sections = "".join(
        f"<figure><img src='{_esc(p.name)}' alt='{_esc(name)}'><figcaption>{_esc(name.replace('_',' ').replace('.png',''))}</figcaption></figure>"
        for name, p in chart_paths.items())
    seeds = summary.get("seeds")
    if not isinstance(seeds, list):
        seeds = sorted({r.get("seed") for r in summary.get("final_runs", []) if isinstance(r, dict) and r.get("seed") is not None})
    policies = summary.get("policies")
    if not isinstance(policies, list):
        policies = [r.get("policy") for r in summary.get("final_summary", []) if isinstance(r, dict) and r.get("policy")]
    limitations = summary.get("limitations", [])
    if not isinstance(limitations, list):
        limitations = [limitations]
    limitation_html = "".join(f"<li>{_esc(v)}</li>" for v in limitations)
    validation = _verification(summary)
    training_facts = _training_facts(summary)
    source_links = []
    for source in summary.get("source", summary.get("sources", [])) if isinstance(summary.get("source", summary.get("sources", [])), list) else []:
        if isinstance(source, dict):
            title, target = source.get("title", "來源"), source.get("url") or source.get("path")
            source_links.append(_relative_link(target, html_path, str(title)) if target and not str(target).startswith("http") else (f"<a href='{_esc(target)}'>{_esc(title)}</a>" if target else _esc(title)))
        else:
            source_links.append(_esc(source))
    checkpoint_text = f"{len(checkpoints)} 個檢查點（{_fmt(min(checkpoints), 0)} 至 {_fmt(max(checkpoints), 0)} episodes）" if checkpoints else "summary 中所列檢查點"
    budget_text = f"每個策略預算 {_fmt(budget, 0)} episodes。" if budget is not None else ""
    chart_note = f"{checkpoint_text}；TRAIN 與 HARD 每個檢查點均使用相同的完整 605 個起點。每 seed 線使用實際完成 episodes；均值線與 CI 依共同的 requested budget 對齊，橫座標取該 budget 各 seed 實際完成 episodes 的平均值。曲線未平滑；CI 反映訓練 seed 間變異，不是場景 CI。{budget_text}"
    page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UEVAVD 五種子策略比較</title>
<style>:root{{--bg:#f2f5f9;--ink:#1e2c3b;--muted:#5e6c7b;--line:#d9e1e9;--accent:#326c9d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"Microsoft JhengHei",sans-serif}}main{{max-width:1220px;margin:auto;padding:28px 18px 56px}}h1{{margin:0 0 4px;font-size:30px}}h2{{font-size:21px;margin:0 0 10px}}section{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}}.lead,.muted,small{{color:var(--muted)}}.lead{{margin-top:0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#f4f7fa}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px}}.facts div{{padding:10px;background:#f6f8fb;border-radius:7px;overflow-wrap:anywhere;min-width:0}}.facts small,.facts strong{{display:block}}.facts strong{{font-size:14px;overflow-wrap:anywhere}}.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,450px),1fr));gap:12px}}figure{{margin:0;padding:10px;border:1px solid var(--line);border-radius:8px;min-width:0}}figure img{{width:100%;height:auto;display:block}}figcaption{{color:var(--muted);font-size:13px}}.note{{background:#f7f9fc;padding:11px;border-radius:7px;color:var(--muted)}}a{{color:var(--accent)}}.notice{{border-left:5px solid #d9a242}}</style><body><main>
<h1>UEVAVD 五種子策略比較</h1><p class="lead">五個訓練種子（{_esc(', '.join(map(str, seeds)) if isinstance(seeds, list) else seeds)}）比較七種策略：{_esc(', '.join(map(str, policies)) if isinstance(policies, list) else policies)}。固定 ResNet50 編碼器、官方預訓練 SAM；SAM 不微調；ResNet18 分類器沿用獨立既有版本。</p>
<section><h2>主要結果｜HARD split</h2><p class="muted">表中為跨種子平均值 ± 樣本標準差；回報另列 Student-t 95% CI。Stationary、Sweep 為確定性參考（單一確定值），Random 以五個 evaluation seed 彙總；只有四個學習策略的訓練 seed 是訓練樣本。</p>{_hard_table(summary)}</section>
<section><h2>學習策略配對差值｜HARD split</h2><p class="muted">同一訓練 seed 的左策略減右策略；括號為跨 seed 配對差值的 Student-t 95% CI。準確率以百分點呈現。GenGap = TRAIN return − HARD return。</p>{_paired_table(summary)}</section>
<section><h2>訓練曲線</h2><p class="note">{chart_note}</p><div class="charts">{curve_sections}</div></section>
<section><h2>訓練預算與驗證</h2><div class="facts">{training_facts}</div><p><b>驗證狀態：</b>{validation}</p></section>
<section><h2>逐 seed 報告與軌跡</h2>{_seed_section(summary, html_path)}</section>
<section class="notice"><h2>協定與限制</h2><ul>{limitation_html or '<li>summary.json 未提供限制說明。</li>'}</ul></section>
<section><h2>原始摘要與來源</h2><p><a href='{_esc(os.path.relpath(summary_path.resolve(), html_path.parent.resolve()).replace(os.sep, "/"))}'>summary.json</a></p><p>{'　·　'.join(source_links) if source_links else '未提供額外來源連結。'}</p></section>
</main></body></html>'''
    html_path.write_text(page, encoding="utf-8")
    return {"html": html_path, **chart_paths}


__all__ = ["render_report"]
