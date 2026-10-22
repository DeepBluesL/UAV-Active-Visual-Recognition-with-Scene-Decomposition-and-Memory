"""Unit tests for the frozen versus end-to-end comparison report."""
from __future__ import annotations

import json
import hashlib
import copy
import tempfile
import unittest
from pathlib import Path

from e2e_map.report import build_summary, seed_stats, write_report


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.frozen_path = self.root / "frozen.json"
        self.historical_path = self.root / "historical.json"
        self.e2e_root = self.root / "e2e"
        self.seeds = (42, 43, 44, 45, 46)
        frozen_runs = []
        historical_runs = []
        frozen_summary = {"seeds": list(self.seeds),
            "protocol": {"requested_training_episodes": 300000},
            "final_summary": [{"split": "hard", "policy": "map", "episodes_per_run": 3}],
            "final_runs": frozen_runs}
        historical_summary = {"seeds": list(self.seeds),
            "protocol": {"requested_training_episodes": 300000},
            "final_summary": [{"split": "hard", "policy": "map", "episodes_per_run": 3}],
            "final_runs": historical_runs}
        for seed in self.seeds:
            metrics = []
            for split, episodes in (("train", 4), ("hard", 3)):
                for variant in ("map", "ibe_map"):
                    metrics.append({"split": split, "policy": variant, "episodes": episodes,
                        "accuracy": .6 + (seed - 42) / 100 + (variant == "ibe_map") * .1,
                        "return_mean": .2 + (variant == "ibe_map") * .02,
                        "distance_m": 10.0, "decisions": 2.0,
                        "success_rate": .6, "stop_rate": .8, "budget_end_rate": .2})
            protocol = {"max_decisions": 5, "gamma": .9, "terminal_reward": .5,
                "step_penalty": .1, "distance_penalty_per_unit": .02,
                "recognition_threshold": .2, "success_mode": "top1_and_true_probability",
                "heuristic_stop_threshold": .9}
            frozen_training = {}
            historical_training = {}
            for variant in ("map", "ibe_map"):
                frozen_config = self._config(seed, variant, protocol)
                frozen_config["encoder"] = "frozen ImageNet ResNet50 V1"
                frozen_training[variant] = {"config": frozen_config}
                old_config = copy.deepcopy(frozen_config)
                old_config["observation_signature"]["preprocess"]["encoder_compute_precision"] = "CUDA float16 autocast"
                old_config["observation_signature"]["preprocess"]["features"] = "pooled float16"
                old_config["dataset_fingerprint"] = "old-fp16-feature-fingerprint"
                historical_training[variant] = {"config": old_config, "completed_episodes": 300020,
                                                "updates": 10, "transitions": 100, "elapsed_seconds": 3.0}
            frozen_runs.append({"seed": seed, "metrics": metrics, "training": frozen_training,
                "results_path": f"frozen/seed_{seed}/results.json",
                "checkpoint_sha256": {"map": f"map-{seed}", "ibe_map": f"ibe-{seed}"}})
            historical_runs.append({"seed": seed, "metrics": copy.deepcopy(metrics),
                "training": historical_training,
                "checkpoint_sha256": {"map": f"old-map-{seed}", "ibe_map": f"old-ibe-{seed}"}})
            for variant in ("map", "ibe_map"):
                base = self.e2e_root / f"seed_{seed}" / variant
                (base / "train").mkdir(parents=True, exist_ok=True)
                (base / "evaluation").mkdir(parents=True, exist_ok=True)
                e2e_config = self._config(seed, variant, protocol)
                e2e_config["encoder"] = "end-to-end ImageNet ResNet50 V1; all backbone convolution and BN affine parameters trainable"
                e2e_config["pretrained_sha256"] = "pretrained-sha"
                e2e_config["dataset_fingerprint"] = "e2e-feature-fingerprint"
                e2e_config["precision"] = "FP32 weights and policy heads; CUDA bfloat16 encoder autocast; FP32 feature boundary; GradScaler initial128"
                e2e_config["observation_signature"]["preprocess"]["encoder_compute_precision"] = "CUDA float16 autocast"
                e2e_config["observation_signature"]["preprocess"]["features"] = "pooled float16"
                training = {"config": e2e_config,
                    "completed_episodes": 300020, "updates": 10}
                (base / "train" / "training.json").write_text(json.dumps(training), encoding="utf-8")
                checkpoint_path = base / "train" / "last.pt"
                checkpoint_path.write_bytes(f"checkpoint-{seed}-{variant}".encode())
                csv_path = base / "evaluation" / "evaluation.csv"
                trace_path = base / "evaluation" / "trajectories.jsonl.gz"
                csv_path.write_bytes(b"csv evidence")
                trace_path.write_bytes(b"trace evidence")
                evaluated_metrics = []
                for split, episodes in (("train", 4), ("hard", 3)):
                    evaluated_metrics.append({"split": split, "policy": variant, "episodes": episodes,
                        "accuracy": .61 + (seed - 42) / 100 + (variant == "ibe_map") * .12,
                        "return_mean": .22 + (variant == "ibe_map") * .03,
                        "distance_m": 9.0, "decisions": 1.9,
                        "success_rate": .61, "stop_rate": .82, "budget_end_rate": .18})
                result = {"protocol": {**protocol, "evaluation_seed": seed,
                    "encoder_precision": "CUDA bfloat16 autocast; FP32 features and policy heads",
                    "checkpoint_sha256": {variant: hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
                    "raw_observation_signature": e2e_config["observation_signature"],
                    "sam_observation_signature": e2e_config["observation_signature"]},
                    "metrics": evaluated_metrics}
                results_path = base / "evaluation" / "results.json"
                results_path.write_text(json.dumps(result), encoding="utf-8")
                expected_count = 4 + 3 * 3
                verification = {"passed": True,
                    "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (results_path, csv_path, trace_path)},
                    "counts": {key: expected_count for key in ("trajectories", "csv_rows", "unique_episode_keys")}}
                (base / "evaluation" / "verification.json").write_text(json.dumps(verification), encoding="utf-8")
        frozen_summary["verification_total"] = {"passed": True, "trajectories": 30}
        self.frozen_path.write_text(json.dumps(frozen_summary), encoding="utf-8")
        frozen_verification = {"passed": True,
            "input_sha256": {str(self.frozen_path.resolve()): hashlib.sha256(self.frozen_path.read_bytes()).hexdigest()}}
        self.frozen_path.with_name("summary_verification.json").write_text(
            json.dumps(frozen_verification), encoding="utf-8")
        self.historical_path.write_text(json.dumps(historical_summary), encoding="utf-8")

    @staticmethod
    def _config(seed, variant, protocol):
        signature = {"pretrained_sha256": "pretrained-sha", "classifier_sha256": "classifier-sha",
            "dataset_path": "dataset", "preprocess": {"kind": "raw" if variant == "map" else "sam",
                "encoder_compute_precision": "CUDA bfloat16 autocast",
                "features": "pooled2048 float32"}}
        return {"seed": seed, "variant": variant, "requested_episodes": 300000,
            "phase": "full", "training_scenes": [1, 2], "validation_scenes": [],
            "protocol": protocol, "learning_rate": .0001, "optimizer": "Adam",
            "batch_size": 256, "environments": 64, "replay_capacity": 100000,
            "learning_starts": 2048, "updates_per_vector_step": 1,
            "target_update_every": 500,
            "epsilon": {"start": 1., "final": .05, "linear_decay_fraction": .7},
            "double_dqn": True, "loss": "smooth_l1", "gradient_clip_norm": 10.,
            "recurrent": variant == "ibe_map", "checkpoint_selection": "final requested training budget",
            "dataset_fingerprint": "dataset-fingerprint", "observation_signature": signature,
            "encoder": "frozen ImageNet ResNet50 V1"}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_matched_deltas_and_difference_in_differences(self) -> None:
        summary = build_summary(self.frozen_path, self.e2e_root, self.seeds)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["run_count"], 10)
        self.assertAlmostEqual(summary["matched_deltas"]["map"]["accuracy"]["mean"], .01)
        self.assertAlmostEqual(summary["matched_deltas"]["ibe_map"]["accuracy"]["mean"], .03)
        self.assertAlmostEqual(summary["difference_in_differences"]["metrics"]["accuracy"]["mean"], .02)
        self.assertEqual(summary["difference_in_differences"]["metrics"]["accuracy"]["n"], 5)

    def test_historical_fp16_is_supplemental_and_does_not_change_primary_statistics(self) -> None:
        primary = build_summary(self.frozen_path, self.e2e_root, self.seeds)
        with_history = build_summary(self.frozen_path, self.e2e_root, self.seeds,
                                     historical_summary_path=self.historical_path)
        self.assertEqual(primary["matched_deltas"], with_history["matched_deltas"])
        self.assertEqual(primary["difference_in_differences"], with_history["difference_in_differences"])
        self.assertIsNotNone(with_history["historical_paired_deltas_e2e_minus_fp16"])
        self.assertEqual(with_history["historical_paired_deltas_e2e_minus_fp16"]["map"]["accuracy"]["n"], 5)
        _, html_path = write_report(with_history, self.root / "historical-report")
        report_html = html_path.read_text(encoding="utf-8")
        self.assertIn("历史冻结 ResNet50（FP16）", report_html)
        self.assertIn("精度与可训练性均不同", report_html)

    def test_rejects_missing_training_record(self) -> None:
        (self.e2e_root / "seed_42" / "map" / "train" / "training.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            build_summary(self.frozen_path, self.e2e_root, self.seeds)

    def test_rejects_partial_training_budget(self) -> None:
        path = self.e2e_root / "seed_42" / "map" / "train" / "training.json"
        training = json.loads(path.read_text(encoding="utf-8"))
        training["completed_episodes"] = 299999
        path.write_text(json.dumps(training), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_summary(self.frozen_path, self.e2e_root, self.seeds)

    def test_rejects_seed_mismatch(self) -> None:
        path = self.e2e_root / "seed_42" / "ibe_map" / "evaluation" / "results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["protocol"]["evaluation_seed"] = 43
        path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_summary(self.frozen_path, self.e2e_root, self.seeds)

    def test_requires_evaluator_precision_contract_including_fp32_boundary(self) -> None:
        path = self.e2e_root / "seed_42" / "ibe_map" / "evaluation" / "results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["protocol"]["encoder_precision"],
                         "CUDA bfloat16 autocast; FP32 features and policy heads")
        result["protocol"]["encoder_precision"] = "CUDA bfloat16 autocast"
        path.write_text(json.dumps(result), encoding="utf-8")
        verification_path = path.with_name("verification.json")
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification["input_sha256"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "encoder_precision"):
            build_summary(self.frozen_path, self.e2e_root, self.seeds)

    def test_rejects_incomplete_metric_rows(self) -> None:
        path = self.e2e_root / "seed_42" / "map" / "evaluation" / "results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        del result["metrics"][0]["stop_rate"]
        path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stop_rate"):
            build_summary(self.frozen_path, self.e2e_root, self.seeds)

    def test_rejects_mismatched_training_protocol(self) -> None:
        path = self.e2e_root / "seed_42" / "ibe_map" / "train" / "training.json"
        training = json.loads(path.read_text(encoding="utf-8"))
        training["config"]["batch_size"] = 128
        path.write_text(json.dumps(training), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "config.batch_size differs"):
            build_summary(self.frozen_path, self.e2e_root, self.seeds)

    def test_requires_verified_matched_control_summary(self) -> None:
        verification_path = self.frozen_path.with_name("summary_verification.json")
        verification_path.write_text(json.dumps({"passed": False}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "summary verification did not pass"):
            build_summary(self.frozen_path, self.e2e_root, self.seeds)

    def test_seed_statistics_and_report_files(self) -> None:
        stats = seed_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 2.0)
        summary = build_summary(self.frozen_path, self.e2e_root, self.seeds)
        json_path, html_path = write_report(summary, self.root / "report")
        self.assertTrue(json_path.is_file())
        report_html = html_path.read_text(encoding="utf-8")
        self.assertIn("差异中的差异", report_html)
        self.assertIn("各组 HARD 指标", report_html)
        self.assertIn("accuracy_return.png", report_html)


if __name__ == "__main__":
    unittest.main()
