"""Fixture-only publication export tests; no experiment or smoke result is used."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from e2e_map.publish_results import export_results
from e2e_map.report import seed_stats


SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("map", "ibe_map")


def _metric(seed: int, variant: str, split: str, frozen: bool) -> dict:
    offset = (seed - 42) * 0.002 + (variant == "ibe_map") * 0.03 + (not frozen) * 0.01
    return {"split": split, "policy": variant, "episodes": 5 if split == "train" else 3,
            "accuracy": 0.6 + offset, "return_mean": 0.2 + offset,
            "distance_m": 10.0 - offset, "decisions": 2.0,
            "success_rate": 0.6 + offset, "stop_rate": 0.8, "budget_end_rate": 0.2,
            "generalization_gap_return": 0.01}


class PublishResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.e2e_root = self.root / "experiment"
        self.destination = self.root / "docs" / "e2e_results"
        self.e2e_root.mkdir()
        (self.e2e_root / "run_manifest.json").write_text("{}", encoding="utf-8")
        frozen_path = self.root / "frozen-summary.json"
        historical_path = self.root / "historical-summary.json"
        frozen_final_runs = []
        report_runs = []
        metric_values = {regime: {variant: {metric: [] for metric in
                         ("accuracy", "return_mean", "distance_m", "decisions", "success_rate",
                          "stop_rate", "budget_end_rate", "generalization_gap_return")}
                         for variant in VARIANTS} for regime in ("e2e", "frozen")}
        did_values = {metric: [] for metric in metric_values["e2e"]["map"]}
        paired_values = {variant: {metric: [] for metric in metric_values["e2e"][variant]}
                         for variant in VARIANTS}

        for seed in SEEDS:
            frozen_metrics = []
            frozen_training = {}
            e2e_blocks = {"map": {}, "ibe_map": {}}
            frozen_blocks = {"map": {}, "ibe_map": {}}
            frozen_hashes = {}
            for variant in VARIANTS:
                config = {"seed": seed, "variant": variant, "requested_episodes": 300000,
                          "phase": "full", "protocol": {"gamma": 0.9}, "recurrent": variant == "ibe_map"}
                frozen_training[variant] = {"config": {**config, "encoder": "frozen"},
                    "completed_episodes": 300020, "updates": 11, "transitions": 100,
                    "elapsed_seconds": 12.5}
                frozen_hashes[variant] = f"frozen-checkpoint-{seed}-{variant}"
                base = self.e2e_root / f"seed_{seed}" / variant
                (base / "train").mkdir(parents=True)
                (base / "evaluation").mkdir()
                checkpoint = base / "train" / "last.pt"
                checkpoint.write_bytes(f"checkpoint-{seed}-{variant}".encode())
                training = {"config": config, "completed_episodes": 300020, "updates": 11,
                            "transitions": 100, "elapsed_seconds": 14.5}
                training_path = base / "train" / "training.json"
                training_path.write_text(json.dumps(training), encoding="utf-8")

                rows_frozen, rows_e2e = {}, {}
                for split in ("train", "hard"):
                    fm, em = _metric(seed, variant, split, True), _metric(seed, variant, split, False)
                    frozen_metrics.append(fm)
                    rows_frozen[split], rows_e2e[split] = fm, em
                    if split == "hard":
                        for metric in metric_values["frozen"][variant]:
                            metric_values["frozen"][variant][metric].append(fm[metric])
                            metric_values["e2e"][variant][metric].append(em[metric])
                            paired_values[variant][metric].append(em[metric] - fm[metric])
                e2e_blocks[variant] = {"metrics": rows_e2e, "config": config,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
                results_path = base / "evaluation" / "results.json"
                results_path.write_text(json.dumps({"metrics": list(rows_e2e.values())}), encoding="utf-8")
                verification = {"passed": True, "counts": {"trajectories": 8,
                    "csv_rows": 8, "unique_episode_keys": 8},
                    "input_sha256": {str(results_path.resolve()): hashlib.sha256(results_path.read_bytes()).hexdigest()}}
                verification_path = base / "evaluation" / "verification.json"
                verification_path.write_text(json.dumps(verification), encoding="utf-8")
                e2e_blocks[variant].update({
                    "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
                    "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
                    "verification_sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest()})
                frozen_blocks[variant] = {"metrics": rows_frozen,
                    "checkpoint_sha256": frozen_hashes[variant]}
            for metric in did_values:
                emap = e2e_blocks["map"]["metrics"]["hard"][metric]
                eibe = e2e_blocks["ibe_map"]["metrics"]["hard"][metric]
                fmap = frozen_blocks["map"]["metrics"]["hard"][metric]
                fibe = frozen_blocks["ibe_map"]["metrics"]["hard"][metric]
                did_values[metric].append((eibe - emap) - (fibe - fmap))
            frozen_final_runs.append({"seed": seed, "metrics": frozen_metrics,
                "training": frozen_training, "checkpoint_sha256": frozen_hashes,
                "verification_counts": {"trajectories": 100, "csv_rows": 100,
                                         "unique_episode_keys": 100}})
            report_runs.append({"seed": seed, "e2e": e2e_blocks, "frozen": frozen_blocks,
                                "historical": frozen_blocks.copy()})

        frozen_source = {"final_runs": frozen_final_runs,
                         "verification_total": {"passed": True, "trajectories": 1000}}
        frozen_path.write_text(json.dumps(frozen_source), encoding="utf-8")
        control_verification = {"passed": True,
            "input_sha256": {str(frozen_path.resolve()): hashlib.sha256(frozen_path.read_bytes()).hexdigest()}}
        (frozen_path.parent / "summary_verification.json").write_text(json.dumps(control_verification), encoding="utf-8")
        historical_path.write_text(json.dumps(frozen_source), encoding="utf-8")
        self.summary = {"status": "complete", "seeds": list(SEEDS), "variants": list(VARIANTS),
            "training_budget_episodes": 300000, "hard_episodes_per_run": 3,
            "e2e_root": str(self.e2e_root.resolve()),
            "frozen_summary_path": str(frozen_path.resolve()),
            "frozen_summary_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "frozen_verification_sha256": hashlib.sha256((frozen_path.parent / "summary_verification.json").read_bytes()).hexdigest(),
            "frozen_verification_total": frozen_source["verification_total"],
            "historical_summary_path": str(historical_path.resolve()),
            "historical_summary_sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
            "final_runs": report_runs,
            "matched_deltas": {variant: {metric: seed_stats(values)
                for metric, values in paired_values[variant].items()} for variant in VARIANTS},
            "historical_paired_deltas_e2e_minus_fp16": {variant: {metric: seed_stats(values)
                for metric, values in paired_values[variant].items()} for variant in VARIANTS},
            "difference_in_differences": {"definition": "e2e IBE-MAP - e2e MAP minus frozen IBE-MAP - frozen MAP",
                "metrics": {metric: seed_stats(values) for metric, values in did_values.items()}},
            "interpretation_limits": ["Fixture export validation only."]}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exports_compact_paths_free_bilingual_artifacts(self) -> None:
        artifacts = export_results(self.summary, self.e2e_root, self.destination)
        self.assertEqual(set(artifacts), {"readme", "metrics", "plot"})
        self.assertTrue(all(path.is_file() for path in artifacts.values()))
        self.assertTrue(artifacts["plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        metrics = json.loads(artifacts["metrics"].read_text(encoding="utf-8"))
        self.assertEqual(len(metrics["e2e_runs"]), 10)
        self.assertEqual(len(metrics["frozen_control_runs"]), 10)
        self.assertEqual(len(metrics["historical_fp16_reference_runs"]), 10)
        self.assertEqual(len(metrics["group_summary"]), 6)
        self.assertEqual(metrics["training_totals_e2e"]["actual_completed_episodes"], 3000200)
        self.assertEqual(metrics["difference_in_differences"]["metrics"]["accuracy"]["n"], 5)
        self.assertNotIn(str(self.e2e_root), artifacts["metrics"].read_text(encoding="utf-8"))
        readme = artifacts["readme"].read_text(encoding="utf-8")
        self.assertIn("## 中文结果", readme)
        self.assertIn("## English brief", readme)
        self.assertIn("accuracy_return.png", readme)
        self.assertIn("区间跨 0", readme)
        self.assertIn("| 策略 | 指标 | E2E BF16 − 历史冻结 FP16", readme)
        self.assertIn("|---|---|---|", readme)
        self.assertIn("SAM 仍使用官方预训练模型", readme)
        self.assertIn("论文未说明其策略 backbone 是否冻结", readme)
        self.assertIn("不能用于单独归因", readme)

    def test_refuses_noncomplete_summary(self) -> None:
        self.summary["status"] = "incomplete"
        with self.assertRaisesRegex(ValueError, "complete"):
            export_results(self.summary, self.e2e_root, self.destination)

    def test_refuses_failed_verification(self) -> None:
        path = self.e2e_root / "seed_42" / "map" / "evaluation" / "verification.json"
        verification = json.loads(path.read_text(encoding="utf-8"))
        verification["passed"] = False
        path.write_text(json.dumps(verification), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification failed"):
            export_results(self.summary, self.e2e_root, self.destination)

    def test_refuses_verification_hash_mismatch(self) -> None:
        path = self.e2e_root / "seed_42" / "map" / "evaluation" / "results.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "results changed|Verified input changed"):
            export_results(self.summary, self.e2e_root, self.destination)

    def test_refuses_changed_matched_control_verification(self) -> None:
        path = Path(self.summary["frozen_summary_path"]).with_name("summary_verification.json")
        verification = json.loads(path.read_text(encoding="utf-8"))
        verification["passed"] = False
        path.write_text(json.dumps(verification), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification source is missing or changed"):
            export_results(self.summary, self.e2e_root, self.destination)


if __name__ == "__main__":
    unittest.main()
