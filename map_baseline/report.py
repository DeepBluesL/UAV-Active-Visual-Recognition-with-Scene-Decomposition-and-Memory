"""Create an offline MAP baseline evaluation report and comparison chart."""

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
from PIL import Image, ImageDraw


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return f"{number:.{digits}f}" if number is not None else "N/A"


def _safe(value: Any, fallback: str = "N/A") -> str:
    return str(value) if value is not None and str(value) else fallback


def _image_data(path_value: Any, bbox: Any = None) -> str | None:
    if not path_value:
        return None
    try:
        path = Path(str(path_value))
        suffix = path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}.get(suffix)
        if not mime:
            return None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                coords = [float(v) for v in bbox]
                if all(math.isfinite(v) for v in coords) and coords[0] != coords[2] and coords[1] != coords[3]:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((min(coords[0], coords[2]), min(coords[1], coords[3]),
                                    max(coords[0], coords[2]), max(coords[1], coords[3])),
                                   outline=(30, 200, 100), width=max(2, image.width // 128))
                    draw.text((max(0, min(coords[0], coords[2])), max(0, min(coords[1], coords[3]) - 16)),
                              "GT bbox", fill=(20, 150, 75))
                    from io import BytesIO
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    return "data:image/png;base64," + encoded
            except (OSError, ValueError, TypeError):
                pass
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return None


def _comparison_chart(metrics: list[dict[str, Any]], output: Path) -> None:
    """Draw one split per row and one metric per column for readable comparisons."""
    usable = [m for m in metrics if isinstance(m, dict)]
    if not usable:
        fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
        ax.axis("off")
        ax.text(0.5, 0.5, "No evaluation metrics provided", ha="center", va="center", fontsize=13)
        fig.savefig(output, dpi=150, facecolor="white")
        plt.close(fig)
        return

    def group_name(metric: dict[str, Any]) -> str:
        split = _safe(metric.get("split"))
        policy = _safe(metric.get("policy"))
        if split == 'hard_fixed_3_moves':
            return 'hard / 3 forced moves'
        if split.casefold() == "hard":
            return "hard fixed-move" if "fixed" in policy.casefold() else "hard native"
        return split

    split_order = list(dict.fromkeys(group_name(m) for m in usable))
    policies = list(dict.fromkeys(_safe(m.get("policy")) for m in usable))
    palette = plt.get_cmap("tab10")
    policy_colors = {policy: palette(i % 10) for i, policy in enumerate(policies)}
    fig, axes = plt.subplots(len(split_order), 3, figsize=(14, 3.0 * len(split_order)),
                             squeeze=False, layout="constrained")
    series = [
        ("accuracy", "Accuracy"),
        ("return_mean", "Mean return"),
        ("distance_m", "Travel distance (m)"),
    ]
    for row, split in enumerate(split_order):
        split_rows = [m for m in usable if group_name(m) == split]
        row_policies = list(dict.fromkeys(_safe(m.get("policy")) for m in split_rows))
        for col, (key, title) in enumerate(series):
            ax = axes[row, col]
            values = []
            for policy in row_policies:
                entry = next(m for m in split_rows if _safe(m.get("policy")) == policy)
                value = _number(entry.get(key))
                if key == "accuracy" and value is not None and 0 <= value <= 1:
                    value *= 100
                values.append(value)
            positions = list(range(len(values)))
            bars = ax.bar(positions, [v if v is not None else 0 for v in values],
                          color=[policy_colors[p] for p in row_policies])
            ax.set_title(title if row == 0 else "")
            ax.set_xticks(positions, row_policies, rotation=25, ha="right", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{split}\nAccuracy (%)")
                ax.set_ylim(0, 108)
            elif col == 1:
                ax.set_ylabel("Return")
            else:
                ax.set_ylabel("Meters")
            if col != 0:
                ax.margins(y=0.16)
            ax.grid(axis="y", alpha=0.25)
            for bar, value in zip(bars, values):
                label = (f"{value:.1f}%" if key == "accuracy" else f"{value:.2f}") if value is not None else "N/A"
                if value is None:
                    bar.set_alpha(0.12)
                y = bar.get_height()
                ax.annotate(label, (bar.get_x() + bar.get_width() / 2, y),
                            xytext=(0, 3 if y >= 0 else -13), textcoords="offset points",
                            ha="center", va="bottom" if y >= 0 else "top", fontsize=8)
    fig.suptitle("UEVAVD - policy comparison by split", fontsize=14)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def _history_section(training: dict[str, Any]) -> str:
    history = training.get("history")
    if isinstance(history, list) and history:
        records = [row for row in history if isinstance(row, dict)]
        if records:
            cols: list[str] = []
            for record in records:
                for key in record:
                    if key not in cols:
                        cols.append(str(key))
            head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
            rows = "".join("<tr>" + "".join(
                f"<td>{html.escape(_safe(record.get(c), ''))}</td>" for c in cols
            ) + "</tr>" for record in records)
            return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"

    history_path = training.get("history_csv") or training.get("history_path")
    if history_path:
        try:
            import csv
            with Path(str(history_path)).open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                return _history_section({"history": rows})
            return f"<p class='muted'>Training history file found but contains no rows: {html.escape(str(history_path))}</p>"
        except OSError:
            return f"<p class='muted'>Training history could not be read: {html.escape(str(history_path))}</p>"
    return "<p class='muted'>No training history supplied.</p>"


def _table(metrics: list[dict[str, Any]], concise: bool = False) -> str:
    if concise:
        columns = ["policy", "accuracy", "return_mean", "distance_m"]
        titles = ["Policy", "Accuracy", "Mean return", "Distance (m)"]
    else:
        columns = ["split", "policy", "episodes", "accuracy", "return_mean", "distance_m", "decisions", "success_rate"]
        titles = ["Split", "Policy", "Episodes", "Accuracy", "Mean return", "Distance (m)", "Decisions", "Success rate"]
    head = "".join(f"<th>{title}</th>" for title in titles)
    rows = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        cells = []
        for key in columns:
            value = metric.get(key)
            if key in ("accuracy", "success_rate") and _number(value) is not None:
                value = f'{float(value)*100:.2f}%'
            elif key in ("return_mean", "distance_m"):
                value = _fmt(value)
            else:
                value = _safe(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    if not rows:
        rows.append(f"<tr><td colspan='{len(columns)}' class='muted'>No evaluation rows supplied.</td></tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _examples_section(examples: list[dict[str, Any]]) -> str:
    cards = []
    for ep in examples:
        if not isinstance(ep, dict):
            continue
        steps = ep.get("steps") if isinstance(ep.get("steps"), list) else []
        first = next((step for step in steps if isinstance(step, dict)), {})
        image = _image_data(first.get("image"), first.get("bbox"))
        visual = f"<img src='{image}' alt='Example result frame'>" if image else "<div class='no-image'>Frame unavailable</div>"
        pos = first.get("position")
        pos_text = ", ".join(_fmt(v, 1) for v in pos[:3]) if isinstance(pos, (list, tuple)) else "N/A"
        cards.append(
            "<article class='example'>" + visual +
            f"<h3>{html.escape(_safe(ep.get('scene')))} · {html.escape(_safe(ep.get('true_label')))}</h3>" +
            f"<p>Policy: {html.escape(_safe(ep.get('policy')))} · Steps: {len(steps)}</p>" +
            f"<p>First prediction: {html.escape(_safe(first.get('predicted_label')))} · Confidence: {html.escape(_fmt(first.get('confidence')))}</p>" +
            f"<p>Position (x, y, z): {html.escape(pos_text)} · Action: {html.escape(_safe(first.get('action')))}</p>" +
            f"<p>Selection note: {html.escape(_safe(ep.get('selection_note'), 'No note supplied.'))}</p></article>"
        )
    return "".join(cards) or "<p class='muted'>No examples supplied.</p>"


def render_report(result_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write report.html and comparison.png from a result JSON file."""
    result_path, out_dir = Path(result_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError):
        result = {}
    if not isinstance(result, dict):
        result = {}

    metrics = result.get("metrics") if isinstance(result.get("metrics"), list) else []
    hard_native = [m for m in metrics if isinstance(m, dict)
                   and _safe(m.get("split"), "").casefold() == "hard"
                   and "fixed" not in _safe(m.get("policy"), "").casefold()]
    protocol = result.get("protocol") if isinstance(result.get("protocol"), dict) else {}
    training = result.get("training") if isinstance(result.get("training"), dict) else {}
    limitations = result.get("limitations") if isinstance(result.get("limitations"), list) else []
    examples = result.get("examples") if isinstance(result.get("examples"), list) else []
    sources = result.get("sources") if isinstance(result.get("sources"), list) else []

    chart_path = out_dir / "comparison.png"
    _comparison_chart(metrics, chart_path)
    chart_data = "data:image/png;base64," + base64.b64encode(chart_path.read_bytes()).decode("ascii")

    protocol_rows = "".join(
        f"<div class='fact'><span>{html.escape(str(key).replace('_', ' ').title())}</span><strong>{html.escape(_safe(value))}</strong></div>"
        for key, value in protocol.items() if not key.endswith(('sha256','fingerprint'))
    ) or "<p class='muted'>No protocol details supplied.</p>"
    limitations_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in limitations) or "<li>No limitations supplied.</li>"
    training_facts = "".join(
        f"<div class='fact'><span>{html.escape(str(key).replace('_', ' ').title())}</span><strong>{html.escape(_safe(value))}</strong></div>"
        for key, value in training.items() if key not in ("history", "history_csv", "history_path")
    ) or "<p class='muted'>No training metadata supplied.</p>"
    history = _history_section(training)
    source_links = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = html.escape(_safe(source.get("title"), "Source"))
        url = str(source.get("url", ""))
        if url.startswith(("https://", "http://")):
            safe_url = html.escape(url, quote=True)
            source_links.append(f"<li><a href='{safe_url}'>{title}</a></li>")
        else:
            source_links.append(f"<li>{title}</li>")
    sources_html = "".join(source_links) or "<li>No sources supplied.</li>"

    report = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAP Baseline Evaluation Report</title>
<style>
:root{{--bg:#f3f6fa;--panel:#fff;--text:#1d2937;--muted:#637184;--accent:#326b9b;--border:#dce3eb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,Segoe UI,Arial,sans-serif}}main{{max-width:1100px;margin:auto;padding:30px 18px 60px}}h1{{font-size:30px;margin:0}}h2{{font-size:21px;margin:0 0 12px}}h3{{font-size:16px;margin:14px 0 5px}}.sub,.muted{{color:var(--muted)}}.panel{{background:var(--panel);border:1px solid var(--border);border-radius:11px;padding:18px;margin:16px 0}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:9px}}.fact{{display:flex;flex-direction:column;padding:11px;background:#f6f8fb;border-radius:7px;overflow-wrap:anywhere}}.fact span{{font-size:12px;color:var(--muted)}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}}th{{background:#f4f7fa}}.chart{{display:block;width:100%;height:auto}}.examples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:13px}}.example{{border:1px solid var(--border);border-radius:9px;padding:12px;min-width:0}}.example img{{width:100%;height:220px;object-fit:contain;background:#111923;border-radius:5px}}.example p{{margin:5px 0;font-size:13px}}.no-image{{height:150px;display:grid;place-items:center;background:#e8edf3;color:var(--muted)}}a{{color:var(--accent)}}
</style></head><body><main>
<h1>MAP Baseline Evaluation Report</h1><div class="sub">UEVAVD stages 1/2 · learned direction, range and STOP · frozen ResNet50 · one training seed. See assumptions below for differences from the paper.</div>
<section class="panel"><h2>Hard split · native policies</h2><p class="muted">25 held-out contexts × all 605 starting views = 15,125 matched episodes per policy. Compare accuracy, return and travel together.</p>{_table(hard_native, concise=True)}</section>
<section class="panel"><details><summary><strong>Evaluation protocol</strong></summary><p>Every policy uses the same starting states, classifier, legal movement grid, reward and maximum five decisions. MAP learns STOP; sweep and random retain their existing confidence-based stopping rule.</p><div class="facts">{protocol_rows}</div></details></section>
<section class="panel"><details><summary><strong>All results by split and policy</strong></summary>{_table(metrics)}</details></section>
<section class="panel"><h2>Comparison chart</h2><img class="chart" src="{chart_data}" alt="Offline chart with one split per row and accuracy, mean return, and distance columns"></section>
<section class="panel"><h2>Training</h2><details><summary><strong>Training configuration and history</strong></summary><div class="facts">{training_facts}</div><div class="history-scroll">{history}</div></details></section>
<section class="panel"><h2>Example episodes</h2><div class="examples">{_examples_section(examples)}</div></section>
<section class="panel"><h2>Limitations and assumptions</h2><ul>{limitations_html}</ul></section>
<section class="panel"><h2>Sources</h2><ul>{sources_html}</ul></section>
</main></body></html>'''
    html_path = out_dir / "report.html"
    html_path.write_text(report, encoding="utf-8")
    return {"html": html_path, "chart": chart_path}


__all__ = ["render_report"]
