"""Render a simplified active-view demo result as PNG, GIF, and offline HTML.

This module intentionally depends only on Python's standard library and Pillow.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FRAME_SIZE = 320
GREEN = (25, 190, 95)
RED = (220, 70, 70)
BG = (20, 25, 34)
CARD_BG = (32, 39, 51)
TEXT = (239, 243, 248)
MUTED = (166, 179, 196)


def _font(size: int = 15) -> ImageFont.ImageFont:
    # Pillow's bundled default font is available without system font installs.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Compatibility with older Pillow releases.
        return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str,
          fill: tuple[int, int, int] = TEXT, size: int = 15) -> None:
    draw.text(xy, value, font=_font(size), fill=fill)


def _safe_str(value: Any, fallback: str = "N/A") -> str:
    return str(value) if value is not None else fallback


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else fallback
    except (TypeError, ValueError):
        return fallback


def _format_number(value: Any, digits: int = 2) -> str:
    try:
        n = float(value)
        return f"{n:.{digits}f}" if math.isfinite(n) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def _format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _format_number(value, 4)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_metric(item) for item in value) + "]"
    return _safe_str(value)


def _load_image(path_value: Any) -> Image.Image:
    path = Path(str(path_value)) if path_value else None
    if path is not None:
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except (OSError, ValueError):
            pass
    # Missing image paths still produce a useful placeholder in all outputs.
    image = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (55, 62, 73))
    draw = ImageDraw.Draw(image)
    _text(draw, (22, 145), "Image unavailable", MUTED, 18)
    return image


def _bbox_for(image: Image.Image, bbox: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        if x1 == x2 or y1 == y2:
            return None
        # Result coordinates are expected in source-image pixels.
        sx, sy = FRAME_SIZE / image.width, FRAME_SIZE / image.height
        return (round(min(x1, x2) * sx), round(min(y1, y2) * sy),
                round(max(x1, x2) * sx), round(max(y1, y2) * sy))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _frame_image(step: dict[str, Any], frame_number: int, total: int,
                 episode: dict[str, Any]) -> Image.Image:
    source = _load_image(step.get("image"))
    box = _bbox_for(source, step.get("bbox"))
    source = source.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS)
    # Keep image content separate from captions; bbox is explicitly identified as GT.
    header_h = 32
    canvas = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE + 144), BG)
    canvas.paste(source, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    if box:
        shifted = (box[0], box[1] + header_h, box[2], box[3] + header_h)
        draw.rectangle(shifted, outline=GREEN, width=3)
        bx, by = shifted[0], max(header_h, shifted[1] - 19)
        draw.rounded_rectangle((bx, by, min(FRAME_SIZE, bx + 75), by + 18), radius=3, fill=GREEN)
        _text(draw, (bx + 4, by + 2), "GT bbox", (5, 30, 18), 12)
    pred = _safe_str(step.get("predicted_label"))
    conf = _format_number(step.get("confidence"), 3)
    position = step.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        pos_text = "pos " + ", ".join(_format_number(v, 1) for v in position[:3])
    else:
        pos_text = "pos N/A"
    episode_title = _safe_str(episode.get("scene"), "Episode")
    _text(draw, (8, 8), f"Simplified active-view demo | {episode_title}"[:50], TEXT, 12)
    _text(draw, (10, FRAME_SIZE + 40), f"Step {frame_number + 1}/{total}   {pos_text}", TEXT, 14)
    _text(draw, (10, FRAME_SIZE + 65), f"Prediction: {pred}", TEXT, 14)
    _text(draw, (10, FRAME_SIZE + 90), f"Confidence: {conf}  |  green box = GT bbox", MUTED, 12)
    action = _safe_str(step.get("action"), "")
    if action:
        _text(draw, (10, FRAME_SIZE + 115), f"Action: {action}"[:43], MUTED, 12)
    return canvas


def _gif_frame(step: dict[str, Any], frame_number: int, total: int,
               episode: dict[str, Any]) -> Image.Image:
    source = _load_image(step.get("image"))
    box = _bbox_for(source, step.get("bbox"))
    source = source.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS)
    frame_width = 480
    image_x, image_y = (frame_width - FRAME_SIZE) // 2, 64
    frame = Image.new("RGB", (frame_width, FRAME_SIZE + 154), BG)
    frame.paste(source, (image_x, image_y))
    draw = ImageDraw.Draw(frame)
    if box:
        draw.rectangle((box[0] + image_x, box[1] + image_y, box[2] + image_x, box[3] + image_y), outline=GREEN, width=3)
    _text(draw, (10, 8), "Simplified active-view demo", TEXT, 16)
    _text(draw, (10, 31), f"Scene: {_safe_str(episode.get('scene'))}"[:55], MUTED, 13)
    label = _safe_str(step.get("predicted_label"))
    position = step.get("position")
    pos_text = ", ".join(_format_number(v, 1) for v in position[:3]) if isinstance(position, (list, tuple)) and len(position) >= 3 else "N/A"
    _text(draw, (10, 415), f"Step {frame_number + 1}/{total} | Prediction: {label} | conf {_format_number(step.get('confidence'), 2)}"[:69], TEXT, 13)
    _text(draw, (10, 437), f"Position (x,y,z): {pos_text} | green box = GT bbox"[:69], MUTED, 12)
    _text(draw, (10, 458), f"Action: {_safe_str(step.get('action'), 'N/A')}"[:69], MUTED, 12)
    return frame


def _as_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + data


def _html_report(data: dict[str, Any]) -> str:
    episodes = data.get("episodes") if isinstance(data.get("episodes"), list) else []
    classes = data.get("classes") if isinstance(data.get("classes"), list) else []
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    payload_episodes: list[dict[str, Any]] = []
    for ep_index, raw_ep in enumerate(episodes):
        ep = raw_ep if isinstance(raw_ep, dict) else {}
        raw_steps = ep.get("steps") if isinstance(ep.get("steps"), list) else []
        steps = []
        for i, raw_step in enumerate(raw_steps):
            step = raw_step if isinstance(raw_step, dict) else {}
            im = _load_image(step.get("image"))
            box = _bbox_for(im, step.get("bbox"))
            im = im.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS)
            # HTML gets a JPEG background; draw the known GT box into the image.
            if box:
                ImageDraw.Draw(im).rectangle(box, outline=GREEN, width=3)
            position = step.get("position")
            probs = step.get("probabilities") if isinstance(step.get("probabilities"), list) else []
            steps.append({
                "image": _as_data_uri(im),
                "bbox": step.get("bbox"),
                "position": list(position[:3]) if isinstance(position, (list, tuple)) else [],
                "predicted_label": _safe_str(step.get("predicted_label")),
                "confidence": _number(step.get("confidence"), 0.0),
                "probabilities": [_number(p, 0.0) for p in probs],
                "action": _safe_str(step.get("action"), ""),
                "index": i,
            })
        payload_episodes.append({
            "scene": _safe_str(ep.get("scene")),
            "true_label": _safe_str(ep.get("true_label")),
            "policy": _safe_str(ep.get("policy")),
            "distance_m": ep.get("distance_m"),
            "initial_correct": ep.get("initial_correct"),
            "final_correct": ep.get("final_correct"),
            "selection_note": _safe_str(ep.get("selection_note"), ""),
            "steps": steps,
        })
    payload = {"episodes": payload_episodes, "classes": [str(c) for c in classes]}
    payload_text = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    method = html.escape(_safe_str(data.get("method")))
    limitations = data.get("limitations") if isinstance(data.get("limitations"), list) else []
    metric_rows = "".join(
        f"<div class='metric'><span>{html.escape(str(k).replace('_', ' '))}</span><strong>{html.escape(_format_metric(v))}</strong></div>"
        for k, v in metrics.items()
    ) or "<p class='muted'>No metrics provided.</p>"
    limitation_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in limitations) or "<li>No limitations provided.</li>"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simplified active-view demo</title>
<style>
:root{{--bg:#141922;--panel:#202733;--text:#eff3f8;--muted:#a6b3c4;--accent:#28c178;--line:#354152}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}}
main{{max-width:1100px;margin:auto;padding:28px 18px 48px}}h1{{font-size:30px;margin:0 0 4px}}h2{{font-size:20px;margin:24px 0 10px}}h3{{margin:0 0 8px}}.muted,.sub{{color:var(--muted)}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}}.metric{{padding:10px;border-radius:8px;background:#171e28;display:flex;flex-direction:column;gap:3px;min-width:0;overflow-wrap:anywhere}}.metric span{{font-size:12px;color:var(--muted)}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}select,button{{background:#171e28;color:var(--text);border:1px solid var(--line);padding:9px 12px;border-radius:7px;font:inherit}}button{{cursor:pointer}}input[type=range]{{width:min(520px,100%);accent-color:var(--accent)}}.viewer{{display:grid;grid-template-columns:minmax(220px,420px) 1fr;gap:18px;align-items:start;margin-top:16px}}.viewer img{{width:100%;height:auto;border-radius:8px;background:#111}}.info{{min-width:0}}.prob{{margin:8px 0}}.prob-head{{display:flex;justify-content:space-between;gap:8px;font-size:13px}}.track{{height:8px;background:#111923;border-radius:8px;overflow:hidden}}.fill{{height:100%;background:var(--accent)}}.note{{border-left:3px solid var(--accent);padding-left:12px;color:var(--muted)}}.trajectory{{width:100%;max-width:720px;height:auto;background:#171e28;border-radius:8px;margin-top:16px}}.trajectory text{{fill:var(--muted);font:12px system-ui,Arial}}
@media(max-width:650px){{.viewer{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Simplified active-view demo</h1><div class="sub">Method: {method}. This report visualizes the supplied results and does not claim a complete IBE-MAP reproduction.</div>
<section class="panel"><h2>Metrics</h2><div class="metrics">{metric_rows}</div></section>
<section class="panel"><h2>Limitations</h2><ul>{limitation_rows}</ul></section>
<section class="panel"><h2>Episodes</h2><div class="controls"><label for="episode">Episode</label><select id="episode"></select><button id="play">Play</button><button id="pause">Pause</button><label id="stepLabel" for="step">Step</label><input id="step" type="range" min="0" max="0" value="0"></div>
<div class="viewer"><img id="frame" alt="Episode frame with known ground-truth bounding box"><div class="info"><h3 id="title"></h3><div id="meta" class="muted"></div><p id="action"></p><div id="selection" class="note"></div><h3>Class probabilities</h3><div id="probabilities"></div></div></div><h3>XY trajectory</h3><svg id="trajectory" class="trajectory" viewBox="0 0 720 210" role="img" aria-label="XY path through visited viewpoints"><line x1="44" y1="176" x2="700" y2="176" stroke="#657386"/><line x1="44" y1="12" x2="44" y2="176" stroke="#657386"/><text x="350" y="202">X</text><text x="8" y="18">Y</text><polyline id="route" fill="none" stroke="#28c178" stroke-width="3"/><g id="points"></g></svg></section>
<p class="muted">The green rectangle is the provided ground-truth bbox. It is displayed for context and is not a model prediction.</p>
</main><script>
const data={payload_text};
const episodeSelect=document.getElementById('episode'), slider=document.getElementById('step');
const cls=data.classes; let epIndex=0,timer=null;
data.episodes.forEach((ep,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{i+1}} · ${{ep.policy}} · ${{ep.scene}} · ${{ep.true_label}}`;episodeSelect.appendChild(o)}});
function render(){{const ep=data.episodes[epIndex]; if(!ep) return; const max=Math.max(0,ep.steps.length-1);slider.max=max;if(+slider.value>max)slider.value=max;
 const s=ep.steps[+slider.value]; document.getElementById('title').textContent=`${{ep.scene}} — ${{ep.true_label}}`;
 document.getElementById('meta').textContent=`Policy: ${{ep.policy}} · Step ${{+slider.value+1}}/${{ep.steps.length}} · Travel: ${{ep.distance_m??'N/A'}} m · Correct: ${{ep.initial_correct??'N/A'}} → ${{ep.final_correct??'N/A'}}`;
 document.getElementById('selection').textContent=ep.selection_note||'No selection note provided.';
 const positions=ep.steps.map(x=>x.position).filter(x=>Array.isArray(x)&&x.length>=2&&x.slice(0,2).every(v=>Number.isFinite(Number(v))));const route=document.getElementById('route'),points=document.getElementById('points');points.replaceChildren();
 if(positions.length){{const xs=positions.map(x=>Number(x[0])),ys=positions.map(x=>Number(x[1])),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const sx=x=>44+(xmax===xmin?328:(x-xmin)/(xmax-xmin)*656),sy=y=>176-(ymax===ymin?82:(y-ymin)/(ymax-ymin)*164);route.setAttribute('points',positions.map(x=>`${{sx(Number(x[0]))}},${{sy(Number(x[1]))}}`).join(' '));positions.forEach((x,i)=>{{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',sx(Number(x[0])));c.setAttribute('cy',sy(Number(x[1])));c.setAttribute('r',i===+slider.value?7:4);c.setAttribute('fill',i===+slider.value?'#ffcf5c':'#28c178');const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=`Step ${{i+1}}: x=${{x[0]}}, y=${{x[1]}}, z=${{x[2]??'N/A'}}`;c.appendChild(title);points.appendChild(c)}})}}else{{route.setAttribute('points','')}}
  const img=document.getElementById('frame');if(s){{img.src=s.image;document.getElementById('action').textContent=`Prediction: ${{s.predicted_label}} · Confidence: ${{Number(s.confidence).toFixed(3)}} · Position: ${{s.position.join(', ')||'N/A'}} · Action: ${{s.action||'N/A'}}`}}else{{img.removeAttribute('src');document.getElementById('action').textContent='No step frames provided.'}}
 const p=document.getElementById('probabilities');p.replaceChildren();if(s){{const n=Math.max(cls.length,s.probabilities.length);for(let i=0;i<n;i++){{const name=cls[i]??`Class ${{i}}`,v=Math.max(0,Math.min(1,s.probabilities[i]??0));const row=document.createElement('div');row.className='prob';row.innerHTML=`<div class="prob-head"><span></span><span></span></div><div class="track"><div class="fill"></div></div>`;row.querySelector('.prob-head span:first-child').textContent=name;row.querySelector('.prob-head span:last-child').textContent=(100*v).toFixed(1)+'%';row.querySelector('.fill').style.width=(100*v)+'%';p.appendChild(row)}}}}
 if(s&&s.predicted_label==='Unknown'){{p.textContent='Target fully occluded. No classification available.'}}
}}
episodeSelect.onchange=()=>{{clearInterval(timer);timer=null;epIndex=+episodeSelect.value;slider.value=0;render()}};slider.oninput=render;
document.getElementById('play').onclick=()=>{{clearInterval(timer);const ep=data.episodes[epIndex];if(ep&&+slider.value>=ep.steps.length-1){{slider.value=0;render()}}timer=setInterval(()=>{{const ep=data.episodes[epIndex];if(!ep||+slider.value>=ep.steps.length-1){{clearInterval(timer);timer=null;return}}slider.value=+slider.value+1;render()}},1000)}};
document.getElementById('pause').onclick=()=>{{clearInterval(timer);timer=null}};
render();
</script></body></html>'''


def render(results_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write first-episode PNG/GIF and an all-episode self-contained HTML report.

    Missing or malformed optional fields are replaced with safe placeholders. The
    function returns a mapping of output names to paths for a calling program.
    """
    results_path = Path(results_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    eps = data.get("episodes") if isinstance(data.get("episodes"), list) else []
    first = eps[0] if eps and isinstance(eps[0], dict) else {}
    raw_steps = first.get("steps") if isinstance(first.get("steps"), list) else []
    steps = [s if isinstance(s, dict) else {} for s in raw_steps]
    if not steps:
        steps = [{"image": None, "predicted_label": "N/A", "confidence": None}]

    # PNG is a single readable horizontal strip with a caption under every frame.
    panels = [_frame_image(step, i, len(steps), first) for i, step in enumerate(steps)]
    strip = Image.new("RGB", (FRAME_SIZE * len(panels), FRAME_SIZE + 144), BG)
    for i, panel in enumerate(panels):
        strip.paste(panel, (i * FRAME_SIZE, 0))
        if i:
            ImageDraw.Draw(strip).line((i * FRAME_SIZE, 0, i * FRAME_SIZE, strip.height), fill=(73, 85, 102), width=2)
    png_path = out_dir / "sequence.png"
    strip.save(png_path, format="PNG", optimize=True)

    gif_frames = [_gif_frame(step, i, len(steps), first) for i, step in enumerate(steps)]
    gif_path = out_dir / "sequence.gif"
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=1000,
                       loop=0, disposal=2, optimize=False)

    html_path = out_dir / "report.html"
    html_path.write_text(_html_report(data), encoding="utf-8")
    return {"png": png_path, "gif": gif_path, "html": html_path}


__all__ = ["render"]
