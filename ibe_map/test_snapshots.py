"""Diagnostic snapshots must not change training trajectories or parameters."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from map_baseline.environment import Dataset
from map_baseline import experiment as map_experiment
from ibe_map import experiment as ibe_experiment


def synthetic_data():
    data = Dataset.__new__(Dataset)
    rng = np.random.default_rng(111)
    data.scenes, data.scene_to_slot = [1], {1: 0}
    data.features = rng.normal(size=(605, 2048)).astype(np.float16)
    data.boxes = rng.uniform(0, 256, size=(605, 4)).astype(np.float32)
    data.visible = np.ones(605, bool)
    data.correct = rng.random(605) > .4
    data.true_probability = np.where(data.correct, .8, .1)
    data.observation_signature = {'preprocess': {}}
    data.fingerprint = 'synthetic-snapshot-test'
    return data


class SnapshotTrainingTests(unittest.TestCase):
    def check_training_unchanged(self, module, variant):
        data = synthetic_data()
        temporary_root = Path(__file__).resolve().parents[1] / 'tmp' / 'snapshot_tests'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(temporary_root.resolve()))
            roots = [Path(folder) / 'without', Path(folder) / 'with']
            saved = []
            for every, out in zip((0, 25), roots):
                args = SimpleNamespace(seed=9, success_mode='top1_and_true_probability',
                    scenes=[1], phase='full', cache=Path(folder), variant=variant,
                    device='cpu', episodes=61, environments=8, replay_capacity=128,
                    batch_size=8, learning_starts=16, updates_per_step=1, target_every=10,
                    log_every=25, out=out, save_every=every)
                with patch.object(module, 'Dataset', return_value=data), \
                        patch('torch.cuda.is_available', return_value=False), \
                        contextlib.redirect_stdout(io.StringIO()):
                    module.train(args)
                saved.append(torch.load(out / 'last.pt', weights_only=True))
            self.assertFalse((roots[0] / 'snapshots').exists())
            self.assertEqual(saved[0]['transitions'], saved[1]['transitions'])
            self.assertEqual(saved[0]['updates'], saved[1]['updates'])
            for key, value in saved[0]['state_dict'].items():
                torch.testing.assert_close(value, saved[1]['state_dict'][key], rtol=0, atol=0)
            files = sorted((roots[1] / 'snapshots').glob('*.pt'))
            self.assertEqual([p.name for p in files],
                ['episodes_000000.pt', 'episodes_000025.pt', 'episodes_000050.pt', 'episodes_000061.pt'])
            for point, path in zip((0, 25, 50, 61), files):
                snapshot = torch.load(path, weights_only=True)
                self.assertGreaterEqual(snapshot['completed_episodes'], point)
                self.assertLess(snapshot['completed_episodes'], point + 8)
            self.assertEqual(torch.load(files[0], weights_only=True)['updates'], 0)
            final = torch.load(files[-1], weights_only=True)
            for key, value in saved[1]['state_dict'].items():
                torch.testing.assert_close(value, final['state_dict'][key], rtol=0, atol=0)

    def test_map_snapshotting_preserves_training(self):
        self.check_training_unchanged(map_experiment, 'map')

    def test_gru_snapshotting_preserves_training(self):
        self.check_training_unchanged(ibe_experiment, 'memo_map')


if __name__ == '__main__':
    unittest.main()
