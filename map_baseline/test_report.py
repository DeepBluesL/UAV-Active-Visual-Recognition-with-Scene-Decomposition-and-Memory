"""Regression checks for baseline report inputs and generated artifacts."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from map_baseline.report import render_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / 'results.json'
        self.output = self.root / 'report'

    def test_missing_results_raise(self):
        with self.assertRaises(FileNotFoundError):
            render_report(self.results, self.output)
        self.assertFalse((self.output / 'report.html').exists())

    def test_invalid_json_raises(self):
        self.results.write_text('{', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            render_report(self.results, self.output)
        self.assertFalse((self.output / 'report.html').exists())

    def test_non_object_results_raise(self):
        self.results.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'must contain an object'):
            render_report(self.results, self.output)
        self.assertFalse((self.output / 'report.html').exists())

    def test_valid_results_generate_report_and_chart(self):
        self.results.write_text(json.dumps({'metrics': [{
            'split': 'hard', 'policy': 'map', 'episodes': 4,
            'accuracy': 0.75, 'return_mean': 0.25, 'distance_m': 10,
            'decisions': 2, 'success_rate': 0.75,
        }]}), encoding='utf-8')
        # Match the evaluator's separate plotting process (PyTorch/OpenMP).
        subprocess.run([sys.executable, '-c',
            'import sys; from map_baseline.report import render_report; render_report(sys.argv[1], sys.argv[2])',
            str(self.results), str(self.output)], check=True)
        report = (self.output / 'report.html').read_text(encoding='utf-8')
        self.assertIn('75.00%', report)
        self.assertIn('data:image/png;base64,', report)
        self.assertTrue((self.output / 'comparison.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__ == '__main__':
    unittest.main()
