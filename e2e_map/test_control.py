"""CPU-only checks for frozen-control paths and summary integrity."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import control as c


class ControlTests(unittest.TestCase):
    def test_cache_layout_and_signature_changes_are_narrow(self):
        root = Path(c.__file__).resolve().parents[1]
        base = root / "output/e2e_multiseed/control"
        raw = c._control_cache(base / "cache", "raw")
        self.assertEqual(raw, base / "cache/raw")
        self.assertEqual(c._scene_paths(raw, 7)[0], base / "cache/raw/scenes/007.npz")
        self.assertEqual(base / "cache/control_cache_manifest.json",
                         (base / "cache" / "control_cache_manifest.json"))
        source = json.loads((root / "output/map/cache/scenes/001.json").read_text(encoding="utf-8"))["signature"]
        changed = c._control_signature(source, "CUDA bfloat16 autocast")
        differences = []

        def visit(left, right, prefix=""):
            if isinstance(left, dict) and isinstance(right, dict):
                for key in left.keys() | right.keys():
                    visit(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else key)
            elif left != right:
                differences.append(prefix)

        visit(source, changed)
        self.assertEqual(set(differences), {
            "preprocess.encoder_compute_precision", "preprocess.features"})

    def test_summary_sidecar_hashes_inputs_and_rejects_damage(self):
        temp_parent = Path(__file__).resolve().parents[1] / "tmp/e2e_control_tests"
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
            root = Path(temporary)
            out = root / "control"
            out.mkdir()
            raw_source, sam_source = root / "source_raw", root / "source_sam"
            weights = root / "weights.bin"
            weights.write_bytes(b"weights")
            source_files = ("source.py",)
            (root / source_files[0]).write_text("locked source", encoding="utf-8")
            seeds = c.SEEDS
            train, simple, hard = c.TRAIN, c.SIMPLE, c.HARD
            # Small scene sets keep this test fast while preserving all four summary groups.
            from contextlib import ExitStack
            with ExitStack() as stack:
                for target, value in (("ROOT", root), ("RAW_SOURCE", raw_source),
                                      ("SAM_SOURCE", sam_source), ("WEIGHTS", weights),
                                      ("SOURCE_FILES", source_files), ("TRAIN", (1,)),
                                      ("SIMPLE", (3,)), ("HARD", (4,))):
                    stack.enter_context(patch.object(c, target, value))

                manifest = {"source_code_sha256": {"source.py": c.sha256(root / "source.py")},
                    "encoder_weights_sha256": c.sha256(weights), "encoder_code_sha256": "encoder-code"}
                (out / "cache").mkdir()
                (out / "cache/control_cache_manifest.json").write_text("{}", encoding="utf-8")
                for kind, source_cache in (("raw", raw_source), ("sam", sam_source)):
                    target_cache = c._control_cache(out / "cache", kind)
                    for scene in (1, 3, 4):
                        for cache in (source_cache, target_cache):
                            npz, metadata = c._scene_paths(cache, scene)
                            npz.parent.mkdir(parents=True, exist_ok=True)
                            npz.write_bytes(f"{kind}-{scene}-npz".encode())
                            metadata.write_text("{}", encoding="utf-8")

                expected_counts = {"trajectories": 2 * c.EXPECTED_EVALUATION_EPISODES,
                    "csv_rows": 2 * c.EXPECTED_EVALUATION_EPISODES,
                    "unique_episode_keys": 2 * c.EXPECTED_EVALUATION_EPISODES,
                    "metrics_groups": 8}
                metric_rows = []
                for split, scene_count in (("train", 1), ("simple", 1), ("hard", 1),
                                           ("hard_fixed_3_moves", 1)):
                    for policy in c.POLICIES:
                        metric_rows.append({"split": split, "policy": policy, "episodes": scene_count * 605,
                            **{metric: 0.5 for metric in c.METRICS}})
                summary_runs = []
                training_totals = {"policy_training_runs": 10, "requested_episodes_per_run": c.EPISODES,
                    "actual_completed_episodes": 0, "transitions": 0, "gradient_updates": 0,
                    "summed_training_seconds": 0.0}
                metric_groups = {}
                verification_total = {key: 0 for key in expected_counts}
                for seed in seeds:
                    seed_root = out / f"seed_{seed}"
                    checkpoints, training_docs = {}, {}
                    for policy in c.POLICIES:
                        job = seed_root / policy
                        (job / "train").mkdir(parents=True)
                        (job / "train/last.pt").write_bytes(f"{seed}-{policy}".encode())
                        training_doc = {"completed_episodes": c.EPISODES, "transitions": 1,
                            "updates": 1, "elapsed_seconds": 1.0}
                        (job / "train/training.json").write_text(json.dumps(training_doc), encoding="utf-8")
                        training_docs[policy] = training_doc
                        checkpoints[policy] = c.sha256(job / "train/last.pt")
                        training_totals["actual_completed_episodes"] += c.EPISODES
                        training_totals["transitions"] += 1
                        training_totals["gradient_updates"] += 1
                        training_totals["summed_training_seconds"] += 1.0
                    evaluation = seed_root / "evaluation"
                    evaluation.mkdir()
                    result = {"protocol": {"evaluation_seed": seed, "checkpoint_sha256": checkpoints},
                        "metrics": [dict(row) for row in metric_rows]}
                    result_path = evaluation / "results.json"
                    result_path.write_text(json.dumps(result), encoding="utf-8")
                    for name in ("evaluation.csv", "trajectories.jsonl.gz"):
                        (evaluation / name).write_bytes(b"fixture")
                    verification = {"passed": True, "counts": expected_counts.copy()}
                    (evaluation / "verification.json").write_text(json.dumps(verification), encoding="utf-8")
                    (evaluation / "postprocess_identity.json").write_text("{}", encoding="utf-8")
                    for row in result["metrics"]:
                        metric_groups.setdefault((row["split"], row["policy"]), []).append(row)
                    for key in verification_total:
                        verification_total[key] += expected_counts[key]
                    summary_runs.append({"seed": seed, "metrics": result["metrics"],
                        "training": training_docs, "checkpoint_sha256": checkpoints,
                        "verification_counts": verification["counts"],
                        "results_path": str(result_path.resolve()),
                        "results_sha256": c.sha256(result_path),
                        "verification_sha256": c.sha256(evaluation / "verification.json"),
                        "postprocess_identity_sha256": c.sha256(evaluation / "postprocess_identity.json")})

                summary = {"seeds": list(seeds), "policies": list(c.POLICIES),
                    "protocol": {"requested_training_episodes": c.EPISODES},
                    "source_hashes": {"control_cache_manifest_sha256": c.sha256(out / "cache/control_cache_manifest.json"),
                        "training_sources": manifest["source_code_sha256"],
                        "encoder_weights_sha256": manifest["encoder_weights_sha256"],
                        "encoder_code_sha256": manifest["encoder_code_sha256"]},
                    "final_runs": summary_runs, "training": training_totals,
                    "final_summary": c._metric_summary(metric_groups),
                    "verification_total": {"passed": True, **verification_total}}
                (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

                def fake_job(_out, seed, policy, _cache):
                    job = _out / f"seed_{seed}/{policy}"
                    result = json.loads((_out / f"seed_{seed}/evaluation/results.json").read_text())
                    training = json.loads((job / "train/training.json").read_text())
                    return training, result, {"checkpoint_sha256": c.sha256(job / "train/last.pt")}

                stack.enter_context(patch.object(c, "_verify_control_cache", return_value=manifest))
                stack.enter_context(patch.object(c, "_evaluation_current", return_value=True))
                stack.enter_context(patch.object(c, "_verify_job_and_evaluation", side_effect=fake_job))
                sidecar = c.verify_summary_control(out)
                summary_hash_path = str((out / "summary.json").resolve())
                self.assertEqual(sidecar["input_sha256"][summary_hash_path], c.sha256(out / "summary.json"))
                training_path = out / "seed_42/map/train/training.json"
                self.assertIn(str(training_path.resolve()), sidecar["input_sha256"])
                training_path.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed after verification"):
                    c.audit_summary_verification(out)


if __name__ == "__main__":
    unittest.main()
