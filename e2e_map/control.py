"""Build and evaluate a matched BF16 frozen-backbone control for E2E MAP."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ibe_map.experiment import validate_cache_for_variant, verify_shared_observations
from ibe_map.multiseed import METRICS, SEEDS, seed_stats
from map_baseline.environment import HARD, SIMPLE, TRAIN, Dataset, Protocol
from map_baseline.features import sha256
from .images import RGBImageStore
from .model import E2EModel, E2ERunner

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
POLICIES = ("map", "ibe_map")
EPISODES = 300_000
MAX_OVERSHOOT = 63
EXPECTED_EVALUATION_EPISODES = len(TRAIN) * 605 + 3 * len(HARD) * 605
RAW_SOURCE = ROOT / "output/map/cache"
SAM_SOURCE = ROOT / "output/ibe_map/cache"
WEIGHTS = RAW_SOURCE / "weights/resnet50-0676ba61.pth"
IMAGE_CACHE = ROOT / "output/e2e_images"
DECOMPOSED = ROOT / "output/ibe_map/decomposed"
SOURCE_FILES = (
    "e2e_map/control.py", "e2e_map/model.py", "e2e_map/images.py",
    "map_baseline/environment.py", "map_baseline/features.py", "map_baseline/model.py",
    "map_baseline/checkpoints.py", "map_baseline/experiment.py",
    "ibe_map/model.py", "ibe_map/experiment.py", "ibe_map/multiseed.py", "ibe_map/verify_results.py",
)
ENCODER_SOURCE_FILES = ("e2e_map/model.py", "e2e_map/images.py", "map_baseline/features.py")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _code_hashes(names=SOURCE_FILES) -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in names}


def _combined_hash(values: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_cache(kind: str) -> Path:
    return RAW_SOURCE if kind == "raw" else SAM_SOURCE


def _control_cache(out: Path, kind: str) -> Path:
    return out / kind


def _scene_paths(cache: Path, scene: int) -> tuple[Path, Path]:
    npz = cache / "scenes" / f"{scene:03d}.npz"
    return npz, npz.with_suffix(".json")


def _control_signature(source_signature: dict, encoder_precision: str) -> dict:
    signature = json.loads(json.dumps(source_signature))
    preprocess = signature["preprocess"]
    preprocess["encoder_compute_precision"] = encoder_precision
    preprocess["features"] = "global average pooled 2048-vector stored as float32"
    return signature


def _valid_control_scene(cache: Path, source_cache: Path, scene: int,
                         source_npz_sha: str, source_meta_sha: str,
                         weights_sha: str, encoder_code_sha: str,
                         expected_signature: dict) -> bool:
    npz, meta = _scene_paths(cache, scene)
    if not npz.is_file() or not meta.is_file():
        return False
    try:
        info = _read_json(meta)
        if not (info.get("npz_sha256") == sha256(npz)
                and info.get("signature") == expected_signature
                and info.get("source_npz_sha256") == source_npz_sha
                and info.get("source_metadata_sha256") == source_meta_sha
                and info.get("encoder_weights_sha256") == weights_sha
                and info.get("encoder_code_sha256") == encoder_code_sha):
            return False
        source_npz, _ = _scene_paths(source_cache, scene)
        with np.load(source_npz, allow_pickle=False) as source, np.load(npz, allow_pickle=False) as target:
            if set(source.files) != set(target.files):
                return False
            if any(not np.array_equal(source[key], target[key])
                   for key in ("probs", "boxes", "visible", "positions", "names")):
                return False
            if target["features"].dtype != np.float32 or target["features"].shape != (605, 2048):
                return False
            if not np.isfinite(target["features"]).all():
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _extract_features(model, runner, slot: int, count: int = 605,
                      batch_size: int = 128) -> np.ndarray:
    output = np.empty((count, 2048), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            ids = slot * 605 + np.arange(start, min(start + batch_size, count))
            output[start:start + len(ids)] = runner.image_features(model, ids).float().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("BF16 encoder produced non-finite float32 features")
    return output


def prepare_control_cache(out: str | Path = ROOT / "output/e2e_multiseed/control/cache",
                          device: str = "cuda") -> dict:
    """Create raw/SAM feature caches using the E2E encoder in BF16, then freeze it."""
    out = Path(out).resolve()
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("BF16 control-cache preparation requires CUDA; no CPU fallback is recorded as BF16")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16 autocast")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Official ImageNet ResNet50 weights are missing: {WEIGHTS}")
    weights_sha = sha256(WEIGHTS)
    if not weights_sha.startswith("0676ba61"):
        raise ValueError("Unexpected ImageNet ResNet50 V1 weight file")
    encoder_code_hashes = _code_hashes(ENCODER_SOURCE_FILES)
    encoder_code_sha = _combined_hash(encoder_code_hashes)
    source_paths = {"raw": RAW_SOURCE, "sam": SAM_SOURCE}
    source_data = {
        "raw": Dataset(RAW_SOURCE, sorted(TRAIN + SIMPLE + HARD)),
        "sam": Dataset(SAM_SOURCE, sorted(TRAIN + SIMPLE + HARD)),
    }
    validate_cache_for_variant(source_data["raw"], "map")
    validate_cache_for_variant(source_data["sam"], "ibe_map")
    verify_shared_observations(source_data["raw"], source_data["sam"])

    all_source_files = {}
    for kind, source_cache in source_paths.items():
        for scene in sorted(TRAIN + SIMPLE + HARD):
            source_npz, source_meta = _scene_paths(source_cache, scene)
            source_info = _read_json(source_meta)
            if source_info.get("npz_sha256") != sha256(source_npz):
                raise ValueError(f"Source observation cache changed: {source_npz}")
            if source_info["signature"]["pretrained_sha256"] != weights_sha:
                raise ValueError(f"Source cache uses a different ResNet50: {source_npz}")
            if source_info["signature"]["preprocess"].get("encoder_compute_precision") != "CUDA float16 autocast":
                raise ValueError(f"Matched control requires the original CUDA float16 cache: {source_meta}")
            all_source_files[f"{kind}/{scene:03d}"] = {
                "npz_sha256": sha256(source_npz),
                "metadata_sha256": sha256(source_meta),
                "signature": source_info["signature"],
            }

    model = E2EModel("map", WEIGHTS, microbatch=64, checkpoint_encoder=False,
                     amp=True, quantize_features=False, encoder_precision="bfloat16")
    model.to(torch.device(device), memory_format=torch.channels_last).eval().requires_grad_(False)
    entries = {"raw": {}, "sam": {}}
    for kind, source_cache in source_paths.items():
        cache = _control_cache(out, kind)
        cache.mkdir(parents=True, exist_ok=True)
        data = source_data[kind]
        variant = "map" if kind == "raw" else "ibe_map"
        images = RGBImageStore(data, ROOT / "dataset", DECOMPOSED, IMAGE_CACHE,
            variant, observation_cache=source_cache)
        runner = E2ERunner(data, images, device)
        old_precision = "CUDA float16 autocast"
        new_precision = "CUDA bfloat16 autocast"
        for slot, scene in enumerate(data.scenes):
            source_npz, source_meta = _scene_paths(source_cache, scene)
            control_npz, control_meta = _scene_paths(cache, scene)
            source_npz_sha, source_meta_sha = sha256(source_npz), sha256(source_meta)
            expected_signature = _control_signature(all_source_files[f"{kind}/{scene:03d}"]["signature"], new_precision)
            # The source may spell precision differently only for an explicitly FP32 CPU cache;
            # this experiment requires the original CUDA-FP16 feature cache as its matched input.
            if _valid_control_scene(cache, source_cache, scene, source_npz_sha, source_meta_sha,
                                    weights_sha, encoder_code_sha, expected_signature):
                entries[kind][f"{scene:03d}"] = {"npz_sha256": sha256(control_npz),
                    "metadata_sha256": sha256(control_meta), "source_npz_sha256": source_npz_sha}
                continue

            with np.load(source_npz, allow_pickle=False) as source:
                arrays = {key: source[key].copy() for key in source.files}
            features = _extract_features(model, runner, slot)
            arrays["features"] = features
            for key in ("probs", "boxes", "visible", "positions", "names"):
                if key not in arrays:
                    raise ValueError(f"Missing immutable {key}: scene {scene}")
            control_npz.parent.mkdir(parents=True, exist_ok=True)
            temp = control_npz.with_suffix(".npz.tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temp.replace(control_npz)
            metadata = {
                "scene": scene, "signature": expected_signature,
                "npz_sha256": sha256(control_npz),
                "source_npz_sha256": source_npz_sha,
                "source_metadata_sha256": source_meta_sha,
                "encoder_weights_sha256": weights_sha,
                "encoder_code_sha256": encoder_code_sha,
                "encoder_code_files": encoder_code_hashes,
                "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
                "precision_note": f"Features recomputed with E2EModel BF16 autocast, float32 output; source precision was {old_precision}.",
            }
            _write_json(control_meta, metadata)
            entries[kind][f"{scene:03d}"] = {"npz_sha256": metadata["npz_sha256"],
                "metadata_sha256": sha256(control_meta), "source_npz_sha256": source_npz_sha}
            print(f"BF16 control {kind} scene {scene:03d}/150", flush=True)

    manifest = {
        "schema_version": 1, "scenes": sorted(TRAIN + SIMPLE + HARD),
        "precision": "CUDA bfloat16 autocast; float32 pooled features",
        "encoder_weights_path": str(WEIGHTS.resolve()), "encoder_weights_sha256": weights_sha,
        "encoder_code_sha256": encoder_code_sha, "encoder_code_files": encoder_code_hashes,
        "source_code_sha256": _code_hashes(), "source_caches": all_source_files,
        "control_caches": entries,
        "raw_observation_signature": source_data["raw"].observation_signature,
        "sam_observation_signature": source_data["sam"].observation_signature,
    }
    _write_json(out / "control_cache_manifest.json", manifest)
    return manifest


def _verify_control_cache(out: Path) -> dict:
    manifest_path = out / "cache/control_cache_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"BF16 control cache manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("scenes") != sorted(TRAIN + SIMPLE + HARD):
        raise ValueError("BF16 control cache does not cover the fixed 150-scene grid")
    if manifest.get("source_code_sha256") != _code_hashes():
        raise ValueError("Control cache source code changed after preparation")
    if manifest.get("encoder_code_sha256") != _combined_hash(_code_hashes(ENCODER_SOURCE_FILES)):
        raise ValueError("Encoder source code changed after preparation")
    if sha256(WEIGHTS) != manifest.get("encoder_weights_sha256"):
        raise ValueError("ImageNet ResNet50 weights changed after cache preparation")
    for kind in ("raw", "sam"):
        source_cache, control_cache = _source_cache(kind), _control_cache(out / "cache", kind)
        source_data = Dataset(source_cache, sorted(TRAIN + SIMPLE + HARD))
        recorded_sig = manifest[f"{kind}_observation_signature"]
        if source_data.observation_signature != recorded_sig:
            raise ValueError(f"Original {kind} observation signature changed")
        for scene in sorted(TRAIN + SIMPLE + HARD):
            source_npz, source_meta = _scene_paths(source_cache, scene)
            cache_npz, cache_meta = _scene_paths(control_cache, scene)
            source_entry = manifest["source_caches"].get(f"{kind}/{scene:03d}")
            target_entry = manifest["control_caches"].get(kind, {}).get(f"{scene:03d}")
            if not source_entry or not target_entry:
                raise ValueError(f"Missing {kind} scene identity {scene:03d}")
            if (sha256(source_npz) != source_entry["npz_sha256"] or
                    sha256(source_meta) != source_entry["metadata_sha256"]):
                raise ValueError(f"Original source cache changed: {source_npz}")
            info = _read_json(cache_meta)
            if (sha256(cache_npz) != info.get("npz_sha256") or
                    sha256(cache_npz) != target_entry["npz_sha256"] or
                    sha256(cache_meta) != target_entry["metadata_sha256"] or
                    info.get("source_npz_sha256") != source_entry["npz_sha256"] or
                    info.get("source_metadata_sha256") != source_entry["metadata_sha256"] or
                    info.get("encoder_weights_sha256") != manifest["encoder_weights_sha256"] or
                    info.get("encoder_code_sha256") != manifest["encoder_code_sha256"]):
                raise ValueError(f"BF16 control cache identity/hash mismatch: {cache_npz}")
            expected_signature = _control_signature(source_entry["signature"], "CUDA bfloat16 autocast")
            if info.get("signature") != expected_signature:
                raise ValueError(f"BF16 control cache signature mismatch: {cache_meta}")
            with np.load(source_npz, allow_pickle=False) as source, np.load(cache_npz, allow_pickle=False) as target:
                for key in ("probs", "boxes", "visible", "positions", "names"):
                    if not np.array_equal(source[key], target[key]):
                        raise ValueError(f"Immutable {key} differs in BF16 control cache: {cache_npz}")
                if target["features"].dtype != np.float32 or target["features"].shape != (605, 2048):
                    raise ValueError(f"Invalid float32 control features: {cache_npz}")
                if not np.isfinite(target["features"]).all():
                    raise ValueError(f"Non-finite control features: {cache_npz}")
    return manifest


def _job_config(cache: Path, seed: int, variant: str, device: str) -> dict:
    data = Dataset(cache, TRAIN)
    config = {"seed": seed, "phase": "full", "requested_episodes": EPISODES,
        "training_scenes": TRAIN, "validation_scenes": [], "protocol": Protocol().as_dict(),
        "learning_rate": 1e-4, "optimizer": "Adam", "batch_size": 256,
        "environments": 64, "replay_capacity": 100000, "learning_starts": 2048,
        "updates_per_vector_step": 1, "target_update_every": 500,
        "epsilon": {"start": 1.0, "final": 0.05, "linear_decay_fraction": 0.7},
        "double_dqn": True, "loss": "smooth_l1", "gradient_clip_norm": 10.0,
        "encoder": "frozen ImageNet ResNet50 V1",
        "dataset_fingerprint": data.fingerprint,
        "observation_signature": data.observation_signature,
        "cache_path": str(cache.resolve()), "device": device}
    if variant == "ibe_map":
        config["variant"] = variant
        config["recurrent"] = True
    return config


def _complete_job(base: Path, seed: int, variant: str, cache: Path, device: str | None) -> bool:
    training_path, checkpoint_path = base / "train/training.json", base / "train/last.pt"
    if not training_path.is_file() or not checkpoint_path.is_file():
        return False
    training = _read_json(training_path)
    config = training.get("config")
    expected = _job_config(cache, seed, variant, device or config.get("device"))
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Wrong BF16 control training identity at {training_path}: {key}")
    completed = training.get("completed_episodes")
    if isinstance(completed, bool) or not isinstance(completed, int):
        return False
    if not EPISODES <= completed <= EPISODES + MAX_OVERSHOOT:
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("config") != config or
            checkpoint.get("completed_episodes") != completed or
            checkpoint.get("updates") != training.get("updates")):
        raise ValueError(f"Training record and final checkpoint disagree: {checkpoint_path}")
    return True


def _run(command: list, logfile: Path) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as log:
        log.write("\nRUN " + json.dumps([str(v) for v in command]) + "\n")
        log.flush()
        subprocess.run([str(v) for v in command], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True,
                       env={**__import__("os").environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})


def _remove_partial_job(base: Path, out: Path, seed: int, variant: str) -> None:
    expected = (out / f"seed_{seed}" / variant).resolve()
    actual = base.resolve()
    if actual != expected or out.resolve() not in actual.parents:
        raise ValueError(f"Refusing to remove path outside this control job: {actual}")
    if actual.exists():
        archive = actual.with_name(f"{actual.name}.partial_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        actual.replace(archive)


def train_control(out: str | Path = ROOT / "output/e2e_multiseed/control",
                  device: str = "cuda") -> list[Path]:
    """Train the ten full-budget frozen controls; incomplete jobs restart cleanly."""
    out = Path(out).resolve()
    manifest = _verify_control_cache(out)
    completed_jobs = []
    for seed in SEEDS:
        for variant in POLICIES:
            base = out / f"seed_{seed}" / variant
            cache = _control_cache(out / "cache", "raw" if variant == "map" else "sam")
            if _complete_job(base, seed, variant, cache, device):
                completed_jobs.append(base)
                continue
            if base.exists():
                _remove_partial_job(base, out, seed, variant)
            train_dir = base / "train"
            if variant == "map":
                command = [sys.executable, "-m", "map_baseline.experiment", "train",
                    "--phase", "full", "--episodes", str(EPISODES), "--seed", str(seed),
                    "--cache", cache, "--out", train_dir, "--device", device,
                    "--batch-size", "256", "--environments", "64", "--replay-capacity", "100000",
                    "--learning-starts", "2048", "--updates-per-step", "1", "--target-every", "500",
                    "--log-every", "5000", "--save-every", "0",
                    "--success-mode", "top1_and_true_probability"]
            else:
                command = [sys.executable, "-m", "ibe_map.experiment", "train",
                    "--variant", "ibe_map", "--phase", "full", "--episodes", str(EPISODES),
                    "--seed", str(seed), "--cache", cache, "--out", train_dir, "--device", device,
                    "--batch-size", "256", "--environments", "64", "--replay-capacity", "100000",
                    "--learning-starts", "2048", "--updates-per-step", "1", "--target-every", "500",
                    "--log-every", "5000", "--save-every", "0",
                    "--success-mode", "top1_and_true_probability"]
            _run(command, base / "training.log")
            if not _complete_job(base, seed, variant, cache, device):
                raise RuntimeError(f"Control training did not reach the complete budget: seed={seed}/{variant}")
            completed_jobs.append(base)
    return completed_jobs


def _evaluation_current(base: Path, seed: int) -> bool:
    result_path = base / "evaluation/results.json"
    verify_path = base / "evaluation/verification.json"
    stamp_path = base / "evaluation/postprocess_identity.json"
    if not all(path.is_file() for path in (result_path, verify_path, stamp_path)):
        return False
    result, verification, stamp = map(_read_json, (result_path, verify_path, stamp_path))
    if result.get("protocol", {}).get("evaluation_seed") != seed:
        return False
    cache_root = base.parent / "cache"
    raw = Dataset(_control_cache(cache_root, "raw"), sorted(TRAIN + SIMPLE + HARD))
    sam = Dataset(_control_cache(cache_root, "sam"), sorted(TRAIN + SIMPLE + HARD))
    protocol = result.get("protocol", {})
    if (protocol.get("raw_cache_fingerprint") != raw.fingerprint
            or protocol.get("sam_cache_fingerprint") != sam.fingerprint
            or protocol.get("raw_observation_signature") != raw.observation_signature
            or protocol.get("sam_observation_signature") != sam.observation_signature):
        return False
    expected_hashes = {p: sha256(base / p / "train/last.pt") for p in POLICIES}
    if result["protocol"].get("checkpoint_sha256") != expected_hashes:
        return False
    if stamp.get("verification_sha256") != sha256(verify_path):
        return False
    if stamp.get("verify_results_sha256") != sha256(ROOT / "ibe_map/verify_results.py"):
        return False
    if verification.get("passed") is not True:
        return False
    counts = verification.get("counts", {})
    if any(counts.get(k) != 2 * EXPECTED_EVALUATION_EPISODES
           for k in ("trajectories", "csv_rows", "unique_episode_keys")):
        return False
    if counts.get("metrics_groups") != 2 * 4 or counts.get("raw_scenes_loaded") != len(TRAIN + SIMPLE + HARD):
        return False
    inputs = verification.get("input_sha256", {})
    for name in ("results.json", "evaluation.csv", "trajectories.jsonl.gz"):
        path = (base / "evaluation" / name).resolve()
        if inputs.get(str(path)) != sha256(path):
            return False
    for path in sorted((_control_cache(base.parent / "cache", "raw") / "scenes").glob("*.npz")):
        if inputs.get(str(path.resolve())) != sha256(path):
            return False
    return True


def evaluate_control(out: str | Path, device: str = "cuda") -> list[Path]:
    """Evaluate both frozen policies on identical starts and independently verify traces."""
    out = Path(out).resolve()
    _verify_control_cache(out)
    for seed in SEEDS:
        for variant in POLICIES:
            base = out / f"seed_{seed}" / variant
            cache = _control_cache(out / "cache", "raw" if variant == "map" else "sam")
            if not _complete_job(base, seed, variant, cache, device):
                raise ValueError(f"All ten BF16 control trainings must be complete before evaluation: {base}")
    completed = []
    raw_cache, sam_cache = _control_cache(out / "cache", "raw"), _control_cache(out / "cache", "sam")
    for seed in SEEDS:
        base = out / f"seed_{seed}"
        evaluation = base / "evaluation"
        if _evaluation_current(base, seed):
            completed.append(evaluation)
            continue
        (evaluation / "verification.json").unlink(missing_ok=True)
        (evaluation / "postprocess_identity.json").unlink(missing_ok=True)
        checkpoints = {p: base / p / "train/last.pt" for p in POLICIES}
        command = [sys.executable, "-m", "ibe_map.experiment", "evaluate",
            "--policies", *POLICIES, "--raw-cache", raw_cache, "--sam-cache", sam_cache,
            "--map-checkpoint", checkpoints["map"], "--ibe-map-checkpoint", checkpoints["ibe_map"],
            "--out", evaluation, "--seed", str(seed), "--device", device, "--no-report"]
        _run(command, base / "evaluation.log")
        verifier_sha = sha256(ROOT / "ibe_map/verify_results.py")
        verifier = [sys.executable, "-m", "ibe_map.verify_results", "--out", evaluation.resolve(),
                    "--raw-cache", raw_cache.resolve(), "--data", (ROOT / "dataset").resolve()]
        _run(verifier, base / "verification.log")
        verify_path = evaluation / "verification.json"
        _write_json(evaluation / "postprocess_identity.json", {
            "verify_results_sha256": verifier_sha,
            "verification_sha256": sha256(verify_path),
            "note": "Independent verifier identity and output hash."})
        if not _evaluation_current(base, seed):
            raise ValueError(f"BF16 control evaluation verification failed: {evaluation}")
        completed.append(evaluation)
    return completed


def _verify_job_and_evaluation(out: Path, seed: int, variant: str, cache: Path) -> tuple[dict, dict, dict]:
    base = out / f"seed_{seed}" / variant
    if not _complete_job(base, seed, variant, cache, None):
        raise ValueError(f"Incomplete or mismatched training run: {base}")
    training = _read_json(base / "train/training.json")
    checkpoint_sha = sha256(base / "train/last.pt")
    result_path = out / f"seed_{seed}/evaluation/results.json"
    result = _read_json(result_path)
    if result.get("protocol", {}).get("evaluation_seed") != seed:
        raise ValueError(f"Evaluation seed mismatch: {result_path}")
    if result["protocol"].get("checkpoint_sha256", {}).get(variant) != checkpoint_sha:
        raise ValueError(f"Evaluated checkpoint differs from training output: {result_path}")
    rows = {(m.get("split"), m.get("policy")): m for m in result.get("metrics", [])}
    if len(rows) != len(result.get("metrics", [])):
        raise ValueError(f"Duplicate metric rows: {result_path}")
    for split in ("train", "simple", "hard", "hard_fixed_3_moves"):
        row = rows.get((split, variant))
        expected = len(TRAIN if split == "train" else SIMPLE if split == "simple" else HARD) * 605
        if not isinstance(row, dict) or row.get("episodes") != expected:
            raise ValueError(f"Incomplete {split}/{variant} evaluation: {result_path}")
    return training, result, {"checkpoint_sha256": checkpoint_sha,
        "results_path": str(result_path.resolve())}


def _metric_summary(metric_groups: dict[tuple[str, str], list[dict]]) -> list[dict]:
    final_summary = []
    for (split, policy), rows in sorted(metric_groups.items()):
        if policy not in POLICIES:
            raise ValueError(f"Unexpected policy in control evaluation: {policy}")
        if len(rows) != len(SEEDS) or len({row["episodes"] for row in rows}) != 1:
            raise ValueError(f"Unmatched control metric group: {split}/{policy}")
        expected_episodes = len(TRAIN if split == "train" else SIMPLE if split == "simple" else HARD) * 605
        if rows[0]["episodes"] != expected_episodes:
            raise ValueError(f"Unexpected episode count in control summary: {split}/{policy}")
        metrics = {name: seed_stats([row[name] for row in rows])
                   for name in METRICS if all(name in row for row in rows)}
        final_summary.append({"split": split, "policy": policy, "randomness": "training_seed",
            "episodes_per_run": rows[0]["episodes"], "metrics": metrics})
    required_groups = {(split, policy) for split in ("train", "simple", "hard") for policy in POLICIES}
    required_groups.add(("hard_fixed_3_moves", "map"))
    required_groups.add(("hard_fixed_3_moves", "ibe_map"))
    if set(metric_groups) != required_groups:
        raise ValueError("Control evaluations are missing or include unexpected policy/split groups")
    return final_summary


def _summary_input_paths(out: Path, destination: Path) -> set[Path]:
    paths: set[Path] = {destination, WEIGHTS, out / "cache/control_cache_manifest.json"}
    paths.update(ROOT / name for name in SOURCE_FILES)
    for kind in ("raw", "sam"):
        source_cache = _source_cache(kind)
        control_cache = _control_cache(out / "cache", kind)
        for scene in sorted(TRAIN + SIMPLE + HARD):
            paths.update(_scene_paths(source_cache, scene))
            paths.update(_scene_paths(control_cache, scene))
    for seed in SEEDS:
        for policy in POLICIES:
            job = out / f"seed_{seed}/{policy}"
            paths.update((job / "train/training.json", job / "train/last.pt"))
        evaluation = out / f"seed_{seed}/evaluation"
        paths.update(evaluation / name for name in (
            "results.json", "evaluation.csv", "trajectories.jsonl.gz", "verification.json",
            "postprocess_identity.json"))
    return paths


def verify_summary_control(out: str | Path) -> dict:
    """Revalidate the written control summary and emit an input-hash sidecar."""
    out = Path(out).resolve()
    destination = out / "summary.json"
    summary = _read_json(destination)
    cache_manifest = _verify_control_cache(out)
    if (summary.get("seeds") != list(SEEDS) or summary.get("policies") != list(POLICIES)
            or summary.get("protocol", {}).get("requested_training_episodes") != EPISODES):
        raise ValueError("Control summary has an unexpected seed, policy, or budget protocol")
    if summary.get("source_hashes") != {
            "control_cache_manifest_sha256": sha256(out / "cache/control_cache_manifest.json"),
            "training_sources": cache_manifest["source_code_sha256"],
            "encoder_weights_sha256": cache_manifest["encoder_weights_sha256"],
            "encoder_code_sha256": cache_manifest["encoder_code_sha256"]}:
        raise ValueError("Control summary source hashes differ from the validated cache manifest")

    metric_groups: dict[tuple[str, str], list[dict]] = {}
    counts = {"training_runs": 0, "evaluated_seeds": 0, "checkpoints": 0,
        "trajectories": 0, "csv_rows": 0, "unique_episode_keys": 0, "metrics_groups": 0}
    input_paths = _summary_input_paths(out, destination)

    runs = summary.get("final_runs")
    if not isinstance(runs, list) or len(runs) != len(SEEDS):
        raise ValueError("Control summary must contain exactly five final runs")
    run_by_seed = {run.get("seed"): run for run in runs if isinstance(run, dict)}
    if len(run_by_seed) != len(SEEDS) or set(run_by_seed) != set(SEEDS):
        raise ValueError("Control summary contains missing or duplicate seeds")
    training_totals = {"policy_training_runs": 10, "requested_episodes_per_run": EPISODES,
        "actual_completed_episodes": 0, "transitions": 0, "gradient_updates": 0,
        "summed_training_seconds": 0.0}
    verification_total = {"trajectories": 0, "csv_rows": 0,
        "unique_episode_keys": 0, "metrics_groups": 0}
    for seed in SEEDS:
        run = run_by_seed[seed]
        results_path = out / f"seed_{seed}/evaluation/results.json"
        if not _evaluation_current(out / f"seed_{seed}", seed):
            raise ValueError(f"Control evaluation is stale or not independently verified: {results_path.parent}")
        result = _read_json(results_path)
        expected_training, expected_checkpoints = {}, {}
        for variant in POLICIES:
            cache = _control_cache(out / "cache", "raw" if variant == "map" else "sam")
            training, _, metadata = _verify_job_and_evaluation(out, seed, variant, cache)
            expected_training[variant] = training
            expected_checkpoints[variant] = metadata["checkpoint_sha256"]
            counts["training_runs"] += 1
            counts["checkpoints"] += 1
            training_totals["actual_completed_episodes"] += training["completed_episodes"]
            training_totals["transitions"] += training["transitions"]
            training_totals["gradient_updates"] += training["updates"]
            training_totals["summed_training_seconds"] += training["elapsed_seconds"]
        evaluation = out / f"seed_{seed}/evaluation"
        verification = _read_json(evaluation / "verification.json")
        counts["evaluated_seeds"] += 1
        for key in verification_total:
            verification_total[key] += verification["counts"][key]
        expected_metrics = result["metrics"]
        expected_run = {"seed": seed, "metrics": expected_metrics,
            "training": expected_training, "checkpoint_sha256": expected_checkpoints,
            "verification_counts": verification["counts"], "results_path": str(results_path.resolve()),
            "results_sha256": sha256(results_path),
            "verification_sha256": sha256(evaluation / "verification.json"),
            "postprocess_identity_sha256": sha256(evaluation / "postprocess_identity.json")}
        if run != expected_run:
            raise ValueError(f"Control summary run record does not match artifacts for seed {seed}")
        for row in expected_metrics:
            metric_groups.setdefault((row["split"], row["policy"]), []).append(row)

    if summary.get("training") != training_totals:
        raise ValueError("Control summary training totals differ from the ten validated jobs")
    if summary.get("final_summary") != _metric_summary(metric_groups):
        raise ValueError("Control summary metrics differ from the five verified evaluations")
    counts.update(verification_total)
    if summary.get("verification_total") != {"passed": True, **verification_total}:
        raise ValueError("Control summary verification counts differ from independently verified runs")
    input_sha256 = {str(path.resolve()): sha256(path) for path in sorted(input_paths)}
    sidecar = {"schema_version": 1, "passed": True,
        "checks": {"all_control_cache_sources_and_arrays": True, "all_training_budgets_and_checkpoints": True,
            "all_independent_trajectory_verifications": True, "summary_recomputed_from_run_artifacts": True,
            "summary_and_source_hashes_recorded": True},
        "counts": counts, "source_hashes": summary["source_hashes"], "input_sha256": input_sha256}
    sidecar_path = out / "summary_verification.json"
    _write_json(sidecar_path, sidecar)
    # Re-read the sidecar and hash its declared inputs once more before reporting success.
    recorded = _read_json(sidecar_path)
    if (recorded.get("passed") is not True or
            recorded.get("input_sha256", {}).get(str(destination.resolve())) != sha256(destination) or
            any(not Path(path).is_file() or sha256(path) != digest
                for path, digest in recorded["input_sha256"].items())):
        raise ValueError("Control summary sidecar failed its post-write integrity check")
    return audit_summary_verification(out)


def audit_summary_verification(out: str | Path) -> dict:
    """Reject a summary or any source artifact that changed after sidecar creation."""
    out = Path(out).resolve()
    destination = out / "summary.json"
    sidecar_path = out / "summary_verification.json"
    summary, sidecar = _read_json(destination), _read_json(sidecar_path)
    if sidecar.get("schema_version") != 1 or sidecar.get("passed") is not True:
        raise ValueError("Control summary verification sidecar is not marked as passed")
    checks = sidecar.get("checks", {})
    expected_checks = {"all_control_cache_sources_and_arrays": True,
        "all_training_budgets_and_checkpoints": True,
        "all_independent_trajectory_verifications": True,
        "summary_recomputed_from_run_artifacts": True,
        "summary_and_source_hashes_recorded": True}
    if checks != expected_checks:
        raise ValueError("Control summary verification is missing a successful integrity check")
    inputs = sidecar.get("input_sha256")
    if not isinstance(inputs, dict) or inputs.get(str(destination.resolve())) != sha256(destination):
        raise ValueError("Control summary sidecar does not cover the current summary")
    expected_inputs = {str(path.resolve()) for path in _summary_input_paths(out, destination)}
    if {str(Path(path).resolve()) for path in inputs} != expected_inputs:
        raise ValueError("Control summary sidecar input inventory is incomplete or unexpected")
    for path, digest in inputs.items():
        file_path = Path(path)
        if not file_path.is_file() or sha256(file_path) != digest:
            raise ValueError(f"Control summary input changed after verification: {file_path}")
    if sidecar.get("source_hashes") != summary.get("source_hashes"):
        raise ValueError("Control summary sidecar source identity differs from summary.json")
    counts = sidecar.get("counts", {})
    expected_counts = {"training_runs": 10, "evaluated_seeds": len(SEEDS),
        "checkpoints": 10, "trajectories": 2 * EXPECTED_EVALUATION_EPISODES * len(SEEDS),
        "csv_rows": 2 * EXPECTED_EVALUATION_EPISODES * len(SEEDS),
        "unique_episode_keys": 2 * EXPECTED_EVALUATION_EPISODES * len(SEEDS),
        "metrics_groups": 8 * len(SEEDS)}
    if counts != expected_counts:
        raise ValueError(f"Control summary verification counts are incomplete: {counts}")
    return sidecar


def summarize_control(out: str | Path) -> Path:
    """Validate ten complete jobs and their two-policy evaluations, then write summary.json."""
    out = Path(out).resolve()
    cache_manifest = _verify_control_cache(out)
    train_cache = {"map": _control_cache(out / "cache", "raw"),
                   "ibe_map": _control_cache(out / "cache", "sam")}
    runs, training_totals = [], {"policy_training_runs": 10,
        "requested_episodes_per_run": EPISODES, "actual_completed_episodes": 0,
        "transitions": 0, "gradient_updates": 0, "summed_training_seconds": 0.0}
    verification_total = {"trajectories": 0, "csv_rows": 0,
        "unique_episode_keys": 0, "metrics_groups": 0}
    metric_groups: dict[tuple[str, str], list[dict]] = {}
    for seed in SEEDS:
        seed_metrics, seed_training, hashes = [], {}, {}
        results_path = out / f"seed_{seed}/evaluation/results.json"
        results = _read_json(results_path)
        if not _evaluation_current(out / f"seed_{seed}", seed):
            raise ValueError(f"Evaluation is incomplete, stale, or not independently verified: {results_path.parent}")
        if results["protocol"].get("checkpoint_sha256") != {
                p: sha256(out / f"seed_{seed}/{p}/train/last.pt") for p in POLICIES}:
            raise ValueError(f"Checkpoint/source identity mismatch: {results_path}")
        for variant in POLICIES:
            training, result, metadata = _verify_job_and_evaluation(
                out, seed, variant, train_cache[variant])
            seed_training[variant] = training
            hashes[variant] = metadata["checkpoint_sha256"]
        seed_metrics = results["metrics"]
        for row in seed_metrics:
            metric_groups.setdefault((row["split"], row["policy"]), []).append(row)
        verification = _read_json(results_path.parent / "verification.json")
        for key in verification_total:
            verification_total[key] += verification["counts"][key]
        for variant in POLICIES:
            record = seed_training[variant]
            training_totals["actual_completed_episodes"] += record["completed_episodes"]
            training_totals["transitions"] += record["transitions"]
            training_totals["gradient_updates"] += record["updates"]
            training_totals["summed_training_seconds"] += record["elapsed_seconds"]
        runs.append({"seed": seed, "metrics": seed_metrics, "training": seed_training,
            "checkpoint_sha256": hashes, "verification_counts": verification["counts"],
            "results_path": str(results_path.resolve()),
            "results_sha256": sha256(results_path),
            "verification_sha256": sha256(results_path.parent / "verification.json"),
            "postprocess_identity_sha256": sha256(results_path.parent / "postprocess_identity.json")})

    final_summary = _metric_summary(metric_groups)
    summary = {
        "schema_version": 1, "seeds": list(SEEDS), "policies": list(POLICIES),
        "learned_policies": list(POLICIES), "training": training_totals,
        "final_runs": runs, "final_summary": final_summary,
        "verification_total": {"passed": True, **verification_total},
        "protocol": {**Protocol().as_dict(), "requested_training_episodes": EPISODES,
            "initial_positions": "all 605 viewpoints per context",
            "control": "frozen ResNet50 features recomputed using the E2E BF16 autocast path; output features float32",
            "verification": "independent trajectory verifier; 211750 trajectories per matched seed"},
        "source_hashes": {"control_cache_manifest_sha256": sha256(out / "cache/control_cache_manifest.json"),
            "training_sources": cache_manifest["source_code_sha256"],
            "encoder_weights_sha256": cache_manifest["encoder_weights_sha256"],
            "encoder_code_sha256": cache_manifest["encoder_code_sha256"]},
    }
    destination = out / "summary.json"
    _write_json(destination, summary)
    verify_summary_control(out)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "summarize"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "output/e2e_multiseed/control")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_control_cache(args.out / "cache", args.device)
    elif args.stage == "train":
        train_control(args.out, args.device)
    elif args.stage == "evaluate":
        evaluate_control(args.out, args.device)
    else:
        print(summarize_control(args.out))


if __name__ == "__main__":
    main()
