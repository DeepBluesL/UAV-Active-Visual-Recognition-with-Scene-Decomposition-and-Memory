"""CPU-only tests for the bounded MAP-only completion/report contract."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import finish_current as f


class FinishCurrentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp")
        self.root = Path(self.temp.name)
        self.config = {
            "seed": 42, "variant": "map", "phase": "full", "requested_episodes": 300000,
            "training_scenes": [1, 2], "validation_scenes": [], "protocol": {"gamma": 0.9},
            "precision": "FP32 weights and policy heads; CUDA bfloat16 encoder autocast; FP32 feature boundary; GradScaler initial128",
        }
        self.training = {
            "config": self.config, "complete": True, "completed_episodes": 300020,
            "updates": 100, "elapsed_seconds": 1234.0,
            "backbone_parameter_l2_change": {"layer4.weight": 0.1},
            "batchnorm_running_stats_unchanged": True,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_validates_exact_full_budget_training_identity(self) -> None:
        config = f.validate_full_training(self.training, plan={
            "training_scenes": [1, 2], "protocol": {"gamma": 0.9}})
        self.assertEqual(config["variant"], "map")
        for mutation, message in (
                ({"completed_episodes": 299999}, "overshoot"),
                ({"complete": False}, "full budget"),
                ({"updates": 0}, "optimizer updates"),
                ({"batchnorm_running_stats_unchanged": False}, "BatchNorm")):
            candidate = {**self.training, **mutation}
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                f.validate_full_training(candidate)

    def test_refuses_wrong_policy_seed_or_precision(self) -> None:
        for key, value in (("variant", "ibe_map"), ("seed", 43), ("precision", "float16")):
            candidate = {**self.training, "config": {**self.config, key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.validate_full_training(candidate)

    def test_build_result_is_map_only_and_has_no_seed_uncertainty(self) -> None:
        splits = ("train", "simple", "hard", "hard_fixed_3_moves")
        metrics = []
        for split in splits:
            metrics.append({"split": split, "policy": "map", "episodes": 3,
                "accuracy": 0.6, "return_mean": 0.2, "distance_m": 10.0,
                "decisions": 2.0, "success_rate": 0.6, "stop_rate": 0.8,
                "budget_end_rate": 0.2})
        e2e = {"metrics": metrics}
        frozen = {"metrics": [{**row, "accuracy": 0.5} for row in metrics]}
        paths = {}
        for label in ("e2e", "frozen"):
            paths[label] = {}
            for item in ("training", "checkpoint", "verification"):
                path = self.root / f"{label}-{item}.bin"
                path.write_bytes(label.encode() + item.encode())
                paths[label][item] = path
        training = {**self.training, "config": self.config}
        verification = {"counts": {key: f.EXPECTED_EVALUATION_EPISODES
                                    for key in ("trajectories", "csv_rows", "unique_episode_keys")}}
        result = f.build_result(e2e, frozen, training, training, verification, verification,
                                {"original_run_manifest_sha256": "manifest",
                                 "original_training_source_sha256": {},
                                 "evaluation_source_sha256": {}},
                                paths["e2e"], paths["frozen"])
        self.assertEqual(result["status"], "complete_single_seed_map_reference")
        self.assertIsNone(result["uncertainty"])
        self.assertIsNone(result["statistical_claims"])
        self.assertAlmostEqual(result["groups"]["hard"]["e2e_minus_frozen"]["accuracy"], 0.1)
        self.assertIn("cannot determine", result["sam_gru_conclusion"])
        self.assertNotIn("ibe_map", json.dumps(result["groups"]))

    def test_readme_explicitly_disclaims_sam_gru_and_ci(self) -> None:
        splits = {split: {"e2e_map": {"accuracy": .6, "return_mean": .2, "distance_m": 10.0},
                          "frozen_bf16_map": {"accuracy": .5, "return_mean": .1, "distance_m": 11.0},
                          "e2e_minus_frozen": {"accuracy": .1, "return_mean": .1, "distance_m": -1.0}}
                  for split in ("train", "simple", "hard", "hard_fixed_3_moves")}
        result = {"groups": splits, "training_runs": [
            {"completed_episodes": 300020}, {"completed_episodes": 300025}],
            "verified_trajectories_each_run": 105875,
            "verified_trajectories_total": 211750}
        path = self.root / "README.md"
        f.write_readme(result, path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("不能根据 MAP-only 结果判断 SAM+GRU", text)
        self.assertIn("不报告跨 seed SD、置信区间、显著性", text)
        self.assertIn("Hard split：固定三步诊断", text)
        self.assertIn("English brief", text)

    def test_scope_manifest_preserves_original_plan_and_source_lock(self) -> None:
        (self.root / "run_manifest.json").write_text(json.dumps({
            "seeds": [42, 43, 44, 45, 46], "variants": ["map", "ibe_map"],
            "episodes": 300000, "source_sha256": {"src.py": hashlib.sha256(b"source").hexdigest()}}),
            encoding="utf-8")
        with patch.object(f.p, "ROOT", self.root), patch.object(f.p, "TRAIN_SOURCES", ("src.py",)), \
             patch.object(f, "SCOPE_SOURCES", ("src.py",)):
            (self.root / "src.py").write_bytes(b"source")
            first = f.scope_identity(self.root)
            second = f.scope_identity(self.root)
            self.assertEqual(first, second)
            self.assertIn("original_run_manifest_sha256", first)
            self.assertEqual(first["policy"], "map")
            (self.root / "src.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source lock changed"):
                f.scope_identity(self.root)


if __name__ == "__main__":
    unittest.main()
