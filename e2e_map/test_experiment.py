"""Execute real training loops to verify that resumable runs remain exact."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from map_baseline.environment import Dataset, STOP
from ibe_map.experiment import PrefixReplay, td_loss as reference_td_loss
from .experiment import train, td_loss
from .model import E2EModel, E2ERunner
from .test_model import ArrayImages, TinyEncoder


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 1, bias=False)

    def forward(self, images):
        return images + torch.relu(self.conv1(images))


class TinyBackbone(nn.Module):
    """Keep stage names so production backbone-change diagnostics also run."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        for number in range(1, 5):
            setattr(self, f'layer{number}', nn.Sequential(TinyBlock()))
        self.projection = nn.Linear(4, 2048)

    def forward(self, images):
        images = torch.relu(self.bn1(self.conv1(images)))
        for number in range(1, 5):
            images = getattr(self, f'layer{number}')(images)
        return self.projection(images.mean(dim=(2, 3)))


def synthetic_data(variant):
    data = Dataset.__new__(Dataset)
    data.scenes, data.scene_to_slot = [1], {1: 0}
    data.boxes = np.full((605, 4), 64, dtype=np.float32)
    data.visible = np.ones(605, dtype=bool)
    data.correct = np.arange(605) % 3 != 0
    data.true_probability = np.where(data.correct, .8, .1)
    data.fingerprint = 'synthetic-605-image-grid'
    data.observation_signature = {'pretrained_sha256': 'test-weights', 'preprocess': {}}
    if variant in ('sam_map', 'ibe_map'):
        data.observation_signature['preprocess']['scene_decomposition'] = {'source': 'test SAM pixels'}
    return data


def tiny_model(variant, weights, **kwargs):
    return E2EModel(variant, encoder=TinyBackbone(), **kwargs)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_nested_equal(self, expected, actual, path=''):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, atol=0, rtol=0, msg=path)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key], f'{path}.{key}')
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_nested_equal(left, right, f'{path}[{index}]')
        else:
            self.assertEqual(expected, actual, path)

    def _run_case(self, variant):
        root = Path(__file__).resolve().parents[1] / 'tmp' / 'e2e_resume_tests'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            base = Path(temporary)
            args = SimpleNamespace(seed=5, scenes=[1], cache=base / 'cache', weights=base / 'weights',
                variant=variant, data=base / 'data', decomposed=base / 'sam', image_cache=base / 'images',
                device='cpu', microbatch=4, no_activation_checkpoint=False, replay_capacity=16,
                environments=4, episodes=19, learning_starts=8, batch_size=4, target_every=2,
                out=base / 'full', save_every=5, log_every=7, resume=False, stop_after_updates=0)
            images = ArrayImages()
            images.fingerprint = 'synthetic-rgb-pixels'
            with ExitStack() as stack:
                stack.enter_context(patch('e2e_map.experiment.Dataset', return_value=synthetic_data(variant)))
                stack.enter_context(patch('e2e_map.experiment.RGBImageStore', return_value=images))
                stack.enter_context(patch('e2e_map.experiment.sha256', return_value='test-weights'))
                stack.enter_context(patch('e2e_map.experiment.E2EModel', side_effect=tiny_model))
                stack.enter_context(patch('torch.cuda.is_available', return_value=False))
                stack.enter_context(redirect_stdout(io.StringIO()))
                full = train(args)
                expected = torch.load(args.out / 'resume.pt', weights_only=True)
                args.out = base / 'resumed'
                args.stop_after_updates = 3
                partial = train(args)
                self.assertFalse(partial['complete'])
                self.assertEqual(partial['updates'], 3)
                saved_snapshots = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (args.out / 'snapshots').glob('*.pt')}
                args.resume = True
                args.stop_after_updates = 4
                second = train(args)
                self.assertEqual(second['updates'], 7)
                args.stop_after_updates = 0
                resumed = train(args)
                actual = torch.load(args.out / 'resume.pt', weights_only=True)
            self.assertTrue(full['complete'])
            self.assertTrue(resumed['complete'])
            # These comparisons cover model/target weights, Adam moments, RNG,
            # partially active episodes and replay after its ring wraps around.
            for key in ('model', 'target', 'optimizer', 'scaler', 'config', 'completed',
                        'transitions', 'updates', 'attempts', 'skipped', 'recent', 'rng',
                        'torch_rng', 'cuda_rng', 'environment', 'replay', 'replay_position', 'replay_size'):
                self.assert_nested_equal(expected[key], actual[key], key)
            self.assert_nested_equal(expected['target_feature_cache'], actual['target_feature_cache'], 'target_feature_cache')
            for name, digest in saved_snapshots.items():
                self.assertEqual(hashlib.sha256((args.out / 'snapshots' / name).read_bytes()).hexdigest(), digest)
            self.assertEqual({p.name for p in (base / 'full' / 'snapshots').glob('*.pt')},
                             {p.name for p in (args.out / 'snapshots').glob('*.pt')})
            for path in (base / 'full' / 'snapshots').glob('*.pt'):
                expected_snapshot = torch.load(path, weights_only=True)
                actual_snapshot = torch.load(args.out / 'snapshots' / path.name, weights_only=True)
                self.assert_nested_equal(expected_snapshot, actual_snapshot, path.name)

    def test_map_interrupted_twice_matches_uninterrupted_training(self):
        self._run_case('map')

    def test_ibe_interrupted_twice_matches_uninterrupted_training(self):
        self._run_case('ibe_map')


class TDLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, terminal):
        data = SimpleNamespace(boxes=np.full((605, 4), 64, dtype=np.float32))
        images = ArrayImages()
        runner = E2ERunner(data, images, 'cpu')
        torch.manual_seed(5)
        model = E2EModel('ibe_map', encoder=TinyEncoder(), microbatch=32,
                         quantize_features=False, amp=False)
        target = copy.deepcopy(model).eval().requires_grad_(False)
        replay = PrefixReplay(4)
        histories = np.array([[1, 2, 3, -1, -1, -1], [4, 5, -1, -1, -1, -1]])
        replay.add(histories, np.array([3, 2]), np.array([STOP, STOP]),
                   np.array([.5, -.5], dtype=np.float32), np.asarray(terminal))
        return runner, model, target, replay

    def test_all_terminal_batch_never_calls_next_network(self):
        runner, model, target, replay = self.fixture([True, True])
        expected = nn.functional.smooth_l1_loss(
            runner.current_q(model, replay.histories[:2], replay.lengths[:2])[:, STOP],
            torch.tensor([.5, -.5]))
        runner.images.calls.clear()
        with patch.object(runner, 'next_q', side_effect=AssertionError('Terminal bootstrap evaluated')):
            actual = td_loss(runner, model, target, replay, np.array([0, 1]), .9)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(set(runner.images.calls), {1, 2, 4})

    def test_mixed_batch_preserves_reference_td_value_without_terminal_next_image(self):
        runner, model, target, replay = self.fixture([True, False])
        sample = np.array([0, 1])
        expected = reference_td_loss(runner, model, target, replay, sample, .9)
        runner.images.calls.clear()
        actual = td_loss(runner, model, target, replay, sample, .9)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(set(runner.images.calls), {1, 2, 4, 5})
        self.assertNotIn(3, runner.images.calls)


if __name__ == '__main__':
    unittest.main()
