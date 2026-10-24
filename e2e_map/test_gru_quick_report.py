"""Small CPU-only rejection tests for the short GRU report audit."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from e2e_map.gru_quick_report import build_report


class QuickReportRejectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        for group in ("e2e", "frozen"):
            base = self.root / "seed_42" / group
            (base / "train").mkdir(parents=True)
            (base / "evaluation").mkdir()
            config = {"variant": "ibe_map", "phase": "full", "seed": 42,
                      "requested_episodes": 5000, "environments": 64}
            (base / "train" / "training.json").write_text(json.dumps({
                "config": config, "completed_episodes": 4999, "updates": 1,
                "elapsed_seconds": 1, "complete": True}), encoding="utf-8")
            (base / "evaluation" / "results.json").write_text("{}", encoding="utf-8")
            (base / "evaluation" / "verification.json").write_text("{}", encoding="utf-8")
            if group == "e2e":
                data = json.loads((base / "train" / "training.json").read_text())
                data["amp_skipped_updates"] = 0
                data["batchnorm_running_stats_unchanged"] = True
                (base / "train" / "training.json").write_text(json.dumps(data), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_short_budget(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            build_report(self.root, "ibe_map", [42], 5000)

    def test_rejects_missing_verification(self):
        for group in ("e2e", "frozen"):
            path = self.root / "seed_42" / group / "train" / "training.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["completed_episodes"] = 5000
            path.write_text(json.dumps(data), encoding="utf-8")
            (self.root / "seed_42" / group / "evaluation" / "verification.json").unlink()
        with self.assertRaisesRegex(ValueError, "verification.json"):
            build_report(self.root, "ibe_map", [42], 5000)


if __name__ == "__main__":
    unittest.main()
