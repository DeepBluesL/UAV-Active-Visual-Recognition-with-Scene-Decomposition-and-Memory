"""Run a small, honest active-view baseline on the actual UEVAVD dataset.

The policy sees only the current observation and its own visited positions.
It never reads future confidence values or the ground-truth class. Ground-truth
boxes are the dataset's supplied target-location input, not detected boxes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

from classifier import CLASSES, load_classifier, predict_images
from render_results import render

ROOT = Path(__file__).resolve().parents[1]
POSITION_RE = re.compile(r"^x_(-?\d+\.\d+)_y_(-?\d+\.\d+)_z_(-?\d+\.\d+)_tar_")
HARD_SCENES = [n for start in (26, 56, 86, 116, 146) for n in range(start, start + 5)]
BOUNDS = ((-50, 50), (-50, 50), (-100, -60))


def read_scene(root: Path, scene: int) -> dict:
    folder = root / f"{scene:02d}"
    boxes = json.loads((folder / "bboxes.json").read_text(encoding="utf-8"))
    views = {}
    # Only annotated images are used. There are a few stray, unannotated PNGs.
    for name, bbox in sorted(boxes.items()):
        match = POSITION_RE.match(name)
        if match is None:
            raise ValueError(f"Unrecognized view filename: {name}")
        pos = tuple(int(float(x)) for x in match.groups())
        if pos in views:
            raise ValueError(f"Duplicate position in scene {scene}: {pos}")
        path = folder / name
        if not path.is_file() or len(bbox) != 4:
            raise ValueError(f"Invalid annotation: {path}")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 <= x2 <= 256 and 0 <= y1 <= y2 <= 256):
            raise ValueError(f"Out of bounds bbox: {path}: {bbox}")
        views[pos] = {"image": str(path.resolve()), "bbox": bbox, "position": list(pos)}
    if len(views) != 605:
        raise ValueError(f"Expected 605 views, got {len(views)} in scene {scene}")
    return views


def scene_label(scene: int) -> str:
    # Dataset block order differs from the alphabetical classifier class order.
    return ["Hatchback", "Pickup", "Sportscar", "Sedan", "SUV"][(scene - 1) // 30]


def infer_scene(views: dict, model, device) -> None:
    positions, crops = [], []
    for pos, view in views.items():
        x1, y1, x2, y2 = view["bbox"]
        view["probabilities"] = []
        view["predicted_label"] = "Unknown"
        view["confidence"] = 0.0
        if x2 <= x1 or y2 <= y1:
            continue
        with Image.open(view["image"]) as image:
            crops.append(image.convert("RGB").crop((x1, y1, x2, y2)))
        positions.append(pos)
    if crops:
        probabilities = predict_images(model, crops, device, batch_size=128)
        if probabilities.shape != (len(crops), len(CLASSES)):
            raise ValueError("Unexpected classifier output shape")
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
            raise ValueError("Invalid classifier probabilities")
        for pos, probs in zip(positions, probabilities):
            prediction = int(np.argmax(probs))
            views[pos].update(probabilities=[float(p) for p in probs],
                              predicted_label=CLASSES[prediction], confidence=float(probs[prediction]))


def scene_fingerprint(views):
    """Invalidate predictions when annotations, image size or mtime change."""
    signature = []
    for pos, view in views.items():
        stat = Path(view["image"]).stat()
        signature.append([pos, view["bbox"], stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def choose_move(position, visited, move_index, policy, rng):
    """Geometry-only movement. No classifier cache or true label is accessible."""
    candidates = []
    for axis, (low, high) in enumerate(BOUNDS):
        for sign in (-1, 1):
            for distance in (10, 20, 30, 40):
                dest = list(position)
                dest[axis] += sign * distance
                dest = tuple(dest)
                if low <= dest[axis] <= high and dest not in visited:
                    candidates.append((dest, axis, sign * distance))
    if not candidates:
        return None
    if policy == "random":
        return rng.choice(candidates)
    # Try a higher view first, then a horizontal sweep. At a boundary, switch
    # axes. This is a heuristic baseline, not the learned IBE-MAP policy.
    axis_order = ([2, 0, 1], [0, 1, 2], [1, 0, 2], [0, 1, 2])[move_index % 4]
    for axis in axis_order:
        sign = -1 if axis == 2 else (1 if position[axis] <= 0 else -1)
        eligible = [c for c in candidates if c[1] == axis and np.sign(c[2]) == sign]
        if eligible:
            return max(eligible, key=lambda c: abs(c[2]))
    return candidates[0]


def rollout(views, scene, start, policy, threshold, max_steps, seed):
    rng = random.Random(seed)
    pos, visited, steps = start, {start}, []
    distance = 0
    for index in range(max_steps):
        current = dict(views[pos])
        steps.append(current)
        if current["confidence"] >= threshold:
            current["action"] = "STOP: confidence threshold"
            break
        if index == max_steps - 1:
            current["action"] = "STOP: observation budget"
            break
        move = choose_move(pos, visited, index, policy, rng)
        if move is None:
            current["action"] = "STOP: no available view"
            break
        dest, axis, delta = move
        current["action"] = f"{'xyz'[axis]} {delta:+d} m"
        assert dest in views and sum(a != b for a, b in zip(pos, dest)) == 1
        assert 10 <= abs(delta) <= 40
        distance += abs(delta)
        pos = dest
        visited.add(pos)
    # Labels are only consulted AFTER the policy has finished.
    truth = scene_label(scene)
    return {"scene": f"{scene:02d}", "true_label": truth, "policy": policy,
            "steps": steps, "distance_m": distance,
            "initial_correct": steps[0]["predicted_label"] == truth,
            "final_correct": steps[-1]["predicted_label"] == truth,
            "selection_note": "Fixed-grid evaluation episode."}


def summarize(episodes):
    return {"episodes": len(episodes),
            "initial_accuracy": float(np.mean([e["initial_correct"] for e in episodes])),
            "final_accuracy": float(np.mean([e["final_correct"] for e in episodes])),
            "mean_distance_m": float(np.mean([e["distance_m"] for e in episodes])),
            "mean_observations": float(np.mean([len(e["steps"]) for e in episodes])),
            "wrong_to_correct": sum(not e["initial_correct"] and e["final_correct"] for e in episodes),
            "correct_to_wrong": sum(e["initial_correct"] and not e["final_correct"] for e in episodes)}


def select_examples(episodes):
    selected = []
    for label in CLASSES:
        examples = [e for e in episodes if e["true_label"] == label]
        success = [e for e in examples if not e["initial_correct"] and e["final_correct"]]
        pool = success or [e for e in examples if len(e["steps"]) > 1] or examples
        if not pool:
            continue
        example = max(pool, key=lambda e: e["steps"][-1]["confidence"] - e["steps"][0]["confidence"])
        example = dict(example)
        example["selection_note"] = (
            "Selected after evaluation to illustrate an incorrect-to-correct transition; "
            "this is not an unbiased performance sample. See all fixed-grid metrics."
            if success else "No wrong-to-correct example found for this class; showing an available episode.")
        selected.append(example)
    selected.sort(key=lambda e: (not e["initial_correct"] and e["final_correct"],
                                e["steps"][-1]["confidence"] - e["steps"][0]["confidence"]), reverse=True)
    failure = next((e for e in episodes if e["initial_correct"] and not e["final_correct"]), None)
    if failure:
        failure = dict(failure)
        failure["selection_note"] = "Failure example: movement changed a correct initial classification into an incorrect one."
        selected.append(failure)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "output/demo")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scenes", type=int, nargs="+", default=HARD_SCENES)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute all classifier predictions")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or not 1 <= args.max_steps <= 5:
        parser.error("threshold must be in (0,1]; max-steps must be 1..5")
    if not args.scenes or any(s not in range(1, 151) for s in args.scenes):
        parser.error("scenes must be nonempty and in 1..150")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.out / "classifier.pt"
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, device = load_classifier(checkpoint, args.device)
    cache_dir = args.out / "cache"
    cache_dir.mkdir(exist_ok=True)
    all_episodes, random_episodes = [], []
    starts = list(itertools.product((-40, 0, 40), (-40, 0, 40), (-60, -80, -100)))
    for scene in args.scenes:
        views = read_scene(args.data / "multiview_imgs", scene)
        dataset_fingerprint = scene_fingerprint(views)
        cache_file = cache_dir / f"scene_{scene:03d}_{fingerprint[:12]}.json"
        cached = json.loads(cache_file.read_text()) if cache_file.exists() and not args.refresh_cache else None
        if cached and cached.get("dataset") == str(args.data.resolve()) and cached.get("scene_fingerprint") == dataset_fingerprint:
            for pos, view in views.items():
                view.update(cached["predictions"][str(pos)])
        else:
            infer_scene(views, model, device)
            predictions = {str(pos): {k: v[k] for k in ("probabilities", "predicted_label", "confidence")}
                           for pos, v in views.items()}
            cache_file.write_text(json.dumps({"dataset": str(args.data.resolve()),
                                             "scene_fingerprint": dataset_fingerprint,
                                             "predictions": predictions}))
        for view in views.values():
            if view["predicted_label"] == "Unknown":
                # Older caches may contain a uniform placeholder; it is not
                # a model prediction and must not be displayed as one.
                view["probabilities"] = []
        for i, start in enumerate(starts):
            seed = args.seed + scene * 1000 + i
            all_episodes.append(rollout(views, scene, start, "confidence-stop sweep", args.threshold, args.max_steps, seed))
            random_episodes.append(rollout(views, scene, start, "random", args.threshold, args.max_steps, seed))
        print(f"Scene {scene:03d}: {len(views)} real views; {len(starts)} matched starts evaluated", flush=True)
    active, baseline = summarize(all_episodes), summarize(random_episodes)
    metrics = {"evaluation_episodes": active["episodes"], "stationary_accuracy": active["initial_accuracy"],
               "sweep_accuracy": active["final_accuracy"], "random_accuracy": baseline["final_accuracy"],
               "sweep_mean_distance_m": active["mean_distance_m"], "random_mean_distance_m": baseline["mean_distance_m"],
               "sweep_mean_observations": active["mean_observations"],
               "sweep_wrong_to_correct": active["wrong_to_correct"], "sweep_correct_to_wrong": active["correct_to_wrong"]}
    result = {
        "method": "ImageNet ResNet18 finetuned on train_605 + confidence-stop axis sweep",
        "classes": CLASSES, "metrics": metrics,
        "limitations": [
            "A runnable effect demonstration, not the paper's full IBE-MAP implementation or reported benchmark.",
            "The linked author repository contains dataset documentation only; no policy code or trained weights were published there at inspection time.",
            "Movement is a deterministic geometric heuristic with confidence-based stopping. SAM, GRU and Dueling DQN are not implemented.",
            "Ground-truth target boxes are supplied to the classifier. This does not demonstrate target localization from scratch.",
            "All model confidences are raw softmax scores and may be overconfident under occlusion.",
            "Evaluation uses 27 fixed starting positions per scene and one seed, not the paper's full training/evaluation protocol.",
            "Displayed successful episodes are selected after evaluation; aggregate metrics include every fixed start and failures.",
            "Images replay recorded simulated views; no Unreal Engine, AirSim runtime or physical UAV is controlled.",
        ],
        "configuration": {"scenes": args.scenes, "starts": starts, "threshold": args.threshold,
                          "max_observations": args.max_steps, "max_movement_m_per_action": 40,
                          "seed": args.seed, "checkpoint_sha256": fingerprint,
                          "coordinate_convention": "AirSim NED: z < 0; altitude = -z in meters",
                          "classifier_input": "supplied visible-target bbox crop, no extra padding"},
        "episodes": select_examples(all_episodes),
        "aggregate": {"sweep": active, "random": baseline},
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["scene", "true_label", "policy", "start_x", "start_y", "start_z", "initial_correct", "final_correct", "distance_m", "observations", "initial_confidence", "final_confidence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in all_episodes + random_episodes:
            row = {k: episode[k] for k in ("scene", "true_label", "policy", "initial_correct", "final_correct", "distance_m")}
            row.update(zip(("start_x", "start_y", "start_z"), episode["steps"][0]["position"]))
            row.update(observations=len(episode["steps"]), initial_confidence=episode["steps"][0]["confidence"], final_confidence=episode["steps"][-1]["confidence"])
            writer.writerow(row)
    outputs = render(args.out / "results.json", args.out)
    print(json.dumps(metrics, indent=2), flush=True)
    print("Artifacts:", {key: str(path.resolve()) for key, path in outputs.items()}, flush=True)


if __name__ == "__main__":
    main()
