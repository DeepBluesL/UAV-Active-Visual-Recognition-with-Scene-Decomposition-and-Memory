"""Build an offline IBE-MAP (SAM+GRU) comparison report.

This module intentionally has no dependency on torch or the policy implementation,
so evaluation workers can generate reports in a small independent process.
"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICY_LABELS = {
    "stationary": "Stationary", "sweep": "Sweep", "random": "Random",
    "map": "MAP", "memo_map": "Memo-MAP (GRU)",
    "sam_map": "SAM-MAP", "ibe_map": "IBE-MAP (SAM+GRU)",
}
POLICY_COLORS = {
    "stationary": "#8c98a4", "sweep": "#4c78a8", "random": "#f2a541",
    "map": "#54a24b", "memo_map": "#b279a2", "sam_map": "#e45756",
    "ibe_map": "#2f7f8f",
}
SPLITS = ("train", "simple", "hard", "hard_fixed_3_moves")


def _num(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, n: int = 3) -> str:
    x = _num(value)
    return f"{x:.{n}f}" if x is not None else "N/A"


def _policy(value: Any) -> str:
    raw = str(value or "N/A")
    return POLICY_LABELS.get(raw, raw)


def _table(rows: list[dict[str, Any]], concise: bool = False) -> str:
    fields = (["policy", "accuracy", "return_mean", "distance_m"] if concise else
              ["split", "policy", "episodes", "accuracy", "return_mean", "distance_m", "decisions"])
    names = {"split": "Split / 数据集", "policy": "Policy / 策略", "episodes": "Episodes / 回合",
             "accuracy": "Accuracy / 准确率", "return_mean": "Mean return / 平均回报",
             "distance_m": "Distance (m) / 距离", "decisions": "Decisions / 决策数"}
    out = ["<div class='scroll'><table><thead><tr>" + "".join(f"<th>{names[f]}</th>" for f in fields) + "</tr></thead><tbody>"]
    if not rows:
        out.append(f"<tr><td colspan='{len(fields)}' class='muted'>No evaluation rows supplied.</td></tr>")
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            if field == "policy": value = _policy(value)
            elif field == "accuracy" and _num(value) is not None: value = f"{float(value)*100:.2f}%"
            elif field in ("return_mean", "distance_m", "decisions"): value = _fmt(value)
            elif field == "split": value = str(value or "N/A").replace("_", " ")
            else: value = str(value if value is not None else "N/A")
            cells.append(f"<td>{html.escape(str(value))}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _chart(metrics: list[dict[str, Any]], path: Path) -> None:
    splits = [s for s in SPLITS if any(m.get("split") == s for m in metrics)]
    splits += [s for s in dict.fromkeys(str(m.get("split")) for m in metrics) if s not in SPLITS]
    if not splits:
        fig, ax = plt.subplots(figsize=(8, 3)); ax.text(.5,.5,"No evaluation metrics provided",ha="center"); ax.axis("off")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); return
    fig, axes = plt.subplots(len(splits), 3, figsize=(15, max(3.3, 2.5*len(splits))), squeeze=False, layout="constrained")
    cols = [("accuracy", "Accuracy (%)"), ("return_mean", "Mean return"), ("distance_m", "Travel distance (m)")]
    for ri, split in enumerate(splits):
        rows = [m for m in metrics if str(m.get("split")) == split]
        policies = [str(m.get("policy", "N/A")) for m in rows]
        for ci, (key, title) in enumerate(cols):
            ax = axes[ri, ci]
            vals = [_num(m.get(key)) for m in rows]
            if key == "accuracy": vals = [v*100 if v is not None and 0 <= v <= 1 else v for v in vals]
            y = list(range(len(rows)))
            colors = [POLICY_COLORS.get(p, plt.get_cmap("tab10")(sum(map(ord, p)) % 10)) for p in policies]
            bars = ax.barh(y, [v if v is not None else 0 for v in vals], color=colors)
            ax.set_yticks(y, [_policy(p) for p in policies], fontsize=8)
            ax.invert_yaxis(); ax.grid(axis="x", alpha=.22); ax.set_axisbelow(True)
            ax.set_title(title if ri == 0 else "")
            if ci == 0:
                ax.set_xlim(0, 108)
            else:
                finite = [v for v in vals if v is not None]
                lo = min([0.0, *finite]) if finite else 0.0
                hi = max([0.0, *finite]) if finite else 1.0
                span = max(hi - lo, abs(hi) * .1, abs(lo) * .1, .1)
                pad = span * .3
                ax.set_xlim(lo - pad, hi + pad)
            xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
            for yi, value in zip(y, vals):
                v = value if value is not None else 0
                offset = max(xspan * .012, .6 if key == "accuracy" else .015)
                ax.text(v + (offset if v >= 0 else -offset), yi,
                        (f"{value:.2f}" if key == "return_mean" else f"{value:.1f}") if value is not None else "N/A", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7)
        axes[ri,0].set_ylabel(split.replace("_", " "))
    fig.suptitle("UEVAVD policy comparison · by split", fontsize=14)
    fig.savefig(path, dpi=160, facecolor="white"); plt.close(fig)


def render_report(results_path: str | Path, out: str | Path) -> dict[str, Path]:
    """Write an independent HTML report and comparison.png; return their paths."""
    results_path, out = Path(results_path), Path(out)
    # Accept either a destination directory or an explicit HTML destination.
    html_path = out if out.suffix.lower() in (".html", ".htm") else out / "report.html"
    out_dir = html_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Results JSON must contain an object: {results_path}")
    metrics = [m for m in data.get("metrics", []) if isinstance(m, dict)] if isinstance(data.get("metrics"), list) else []
    hard = [m for m in metrics if m.get("split") == "hard"]
    chart_path = out_dir / "comparison.png"
    _chart(metrics, chart_path)
    chart_data = "data:image/png;base64," + base64.b64encode(chart_path.read_bytes()).decode("ascii")

    def facts(obj: Any) -> str:
        if not isinstance(obj, dict) or not obj: return "<p class='muted'>No metadata supplied.</p>"
        return "<div class='facts'>" + "".join(
            f"<div><small>{html.escape(str(k).replace('_',' ').title())}</small><strong>{html.escape(str(v))}</strong></div>"
            for k,v in obj.items() if not str(k).endswith(("sha256", "fingerprint"))) + "</div>"
    limitations = data.get("limitations") if isinstance(data.get("limitations"), list) else []
    salient_terms = ("sam", "frozen", "seed")
    salient_limitations = [str(item) for item in limitations
                           if any(term in str(item).casefold() for term in salient_terms)]
    salient_html = "".join(f"<li>{html.escape(item)}</li>" for item in salient_limitations)
    if not salient_html:
        salient_html = "<li>No SAM, encoder-freezing, or seed limitation was specified in the results.</li>"
    lim_html = "".join(f"<li>{html.escape(str(v))}</li>" for v in limitations) or "<li>No limitations supplied.</li>"
    raw_link = html.escape(results_path.resolve().as_uri(), quote=True)
    ex_link = data.get("examples_report") or data.get("examples_dir_report") or "examples/report.html"
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    source_html = "".join("<li>" + (f"<a href='{html.escape(str(s.get('url')), quote=True)}'>{html.escape(str(s.get('title','Source')))}</a>" if isinstance(s,dict) and str(s.get('url','')).startswith(("http://","https://")) else html.escape(str(s))) + "</li>" for s in sources) or "<li>No sources supplied.</li>"
    preview_paths = sorted((out_dir / 'sam_preview').glob('[0-9]*.png'))
    ablation_html = ""
    if results_path.with_name('evaluation.csv').exists():
        from .ablation import analyze
        comparisons = analyze(results_path,out_dir)['data']['comparisons']
        if comparisons:
            ablation_html = "<section><h2>模組差值｜同場景配對比較</h2><p class='muted'>左側策略減右側策略；括號為場景 bootstrap 95% 區間。準確率用百分點，距離用米。這些區間不包含訓練種子的波動。</p><div class='scroll'><table><thead><tr><th>比較</th><th>Δ 準確率</th><th>Δ 回報</th><th>Δ 距離</th></tr></thead><tbody>"
            for row in comparisons:
                if not row['protocol_valid']:
                    raise ValueError(f"Ablation starts mismatch: {row['label']}")
                name = _policy(row['left_policy'])+' − '+_policy(row['right_policy'])
                cells = []
                for metric,digits in [('accuracy',2),('return_mean',4),('distance_m',2)]:
                    v = row['metrics'][metric]
                    cells.append(f"<td>{v['difference']:+.{digits}f} [{v['ci95'][0]:+.{digits}f}, {v['ci95'][1]:+.{digits}f}]</td>")
                ablation_html += '<tr><td>'+html.escape(name)+'</td>'+''.join(cells)+'</tr>'
            ablation_html += '</tbody></table></div></section>'
    preview_html = ""
    if preview_paths:
        preview_html = "<section><h2>SAM 實際分解預覽</h2><p class='muted'>每張由左至右為原圖與提示、SAM 目標掩碼、策略接收的分解圖。識別分類器始終使用原始圖像。</p>"
        for path in preview_paths:
            preview_html += f"<figure><img class='chart' loading='lazy' src='sam_preview/{html.escape(path.name, quote=True)}' alt='SAM decomposition {html.escape(path.stem)}'><figcaption>{html.escape(path.stem)}</figcaption></figure>"
        preview_html += "</section>"
    report = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IBE-MAP Policy Comparison</title>
<style>:root{{--bg:#f3f6fa;--ink:#1d2937;--muted:#637184;--line:#dce3eb;--blue:#326b9b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Microsoft JhengHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:28px 18px 60px}}h1{{margin:0}}.muted,small{{color:var(--muted)}}section{{background:white;border:1px solid var(--line);border-radius:11px;padding:18px;margin:16px 0}}.notice{{border-left:5px solid #e0a13d;background:#fffaf0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#f4f7fa}}.chart{{width:100%;height:auto}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}}.facts div{{padding:10px;background:#f6f8fb;border-radius:7px;overflow-wrap:anywhere}}.facts small,.facts strong{{display:block}}a{{color:var(--blue)}}</style><body><main>
<h1>IBE-MAP policy comparison｜策略比較</h1><p class="muted">MAP、Memo-MAP (GRU)、SAM-MAP、IBE-MAP (SAM+GRU) 與既有基線。</p>
<section class="notice"><h2>Key limitations｜主要限制</h2><ul>{salient_html}</ul></section>
<section><h2>Hard split｜主要比較</h2><p class="muted">主要表格優先呈現 hard split。準確率、回報與移動距離需合併判讀。</p>{_table(hard, True)}</section>
<section><h2>Comparison chart｜比較圖</h2><img class="chart" src="{chart_data}" alt="各 split 的 accuracy、return、distance 橫向長條圖"></section>{ablation_html}
<section><details><summary><strong>All splits｜所有資料切分與策略</strong></summary>{_table(metrics)}</details></section>
<section><details><summary><strong>Protocol｜評估流程</strong></summary>{facts(data.get('protocol'))}</details></section>
<section><details><summary><strong>Training｜訓練資訊</strong></summary>{facts(data.get('training'))}</details></section>
<section><details><summary><strong>Limitations｜限制</strong></summary><ul>{lim_html}</ul></details></section>
<section><h2>Raw metrics｜原始指標</h2><a href="{raw_link}">Open results JSON</a></section>
<section><h2>Examples｜示例</h2><a href="{html.escape(str(ex_link), quote=True)}">Open example report</a></section>{preview_html}
<section><h2>Sources｜來源</h2><ul>{source_html}</ul></section></main></body></html>'''
    html_path.write_text(report, encoding="utf-8")
    return {"html": html_path, "chart": chart_path}


__all__ = ["render_report"]
