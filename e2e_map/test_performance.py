"""CPU checks of the optional performance path; never allocate CUDA tensors."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .benchmark import compare_correctness, make_fixture, validate_input, validate_limits
from .model import E2EModel, E2ERunner
from .performance import (PerformanceModel, PerformanceOptions, PerformanceRunner,
                          TensorPixelStore, runtime_adapters, train)
from .test_model import ArrayImages


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(4, 2048, 1), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        return self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(x))))).flatten(1)


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_checkpoint_modes_preserve_loss_all_layer_gradients_and_bn_statistics(self):
        torch.manual_seed(4)
        reference = E2EModel('map', encoder=SmallResNet(), microbatch=2, amp=False)
        images = torch.randn(5, 3, 8, 8)
        boxes = torch.rand(5, 4) * 100
        reference.train()
        expected = reference(reference.encode(images), boxes)
        expected.square().sum().backward()
        for mode in ('full', 'none', 'late'):
            for microbatch in (2, 4):
                model = PerformanceModel('map', encoder=SmallResNet(), checkpoint_mode=mode,
                                          microbatch=microbatch, amp=False)
                model.load_state_dict(reference.state_dict())
                model.train()
                actual = model(model.encode(images), boxes)
                actual.square().sum().backward()
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                for name, parameter in model.named_parameters():
                    expected_gradient = dict(reference.named_parameters())[name].grad
                    if expected_gradient is not None:
                        torch.testing.assert_close(parameter.grad, expected_gradient, atol=2e-6, rtol=2e-4)
                self.assertFalse(model.encoder.bn1.training)
                torch.testing.assert_close(model.encoder.bn1.running_mean, reference.encoder.bn1.running_mean,
                                           atol=0, rtol=0)

    def test_cached_pixels_are_exact_immutable_and_do_not_cache_neural_features(self):
        source = ArrayImages()
        original = source.images.copy()
        cache = TensorPixelStore(source, len(original), 'cpu', load_batch=128)
        ids = np.array([604, 0, 12, 12, 100])
        np.testing.assert_array_equal(cache.get(ids).numpy(), original[ids])
        source.images[:] = 0
        np.testing.assert_array_equal(cache.get(ids).numpy(), original[ids])
        self.assertEqual(cache.bytes, original.nbytes)
        with self.assertRaisesRegex(ValueError, 'outside'):
            cache.get([-1])
        with self.assertRaisesRegex(ValueError, 'outside'):
            cache.get([605])

    def test_pixel_cache_runner_matches_existing_observations_and_recomputes_after_update(self):
        images = ArrayImages()
        data = SimpleNamespace(boxes=np.full((605, 4), 64, np.float32))
        original = E2ERunner(data, images, 'cpu')
        cached = PerformanceRunner(data, images, 'cpu', gpu_pixels=True)
        model = E2EModel('map', encoder=SmallResNet(), amp=False).eval()
        history = np.array([[1, 2], [4, 5]])
        lengths = np.array([2, 2])
        expected = original.current_q(model, history, lengths)
        actual = cached.current_q(model, history, lengths)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with torch.no_grad():
            model.encoder.conv1.weight.add_(.1)
        changed = cached.current_q(model, history, lengths)
        self.assertGreater(float((changed - actual).detach().abs().max()), 1e-7)

    def test_runtime_adapters_restore_original_functions_even_on_failure(self):
        from . import experiment
        originals = experiment.E2EModel, experiment.E2ERunner, experiment.atomic_save
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with runtime_adapters(PerformanceOptions('none', 128), {'options': 'test'}):
                self.assertNotEqual(experiment.E2EModel, originals[0])
                raise RuntimeError('deliberate')
        self.assertEqual((experiment.E2EModel, experiment.E2ERunner, experiment.atomic_save), originals)

    def test_checkpoint_adapter_embeds_performance_without_changing_original_config(self):
        from . import experiment
        config = {'batch_size': 256, 'protocol': 'unchanged'}
        saved = {}
        def capture(value, path):
            saved.update(value)
        with patch.object(experiment, 'atomic_save', side_effect=capture):
            with runtime_adapters(PerformanceOptions('none', 128), {'marker': 'runtime identity'}):
                experiment.atomic_save({'config': config, 'state_dict': {}}, Path('unused'))
        self.assertEqual(saved['config'], config)
        self.assertEqual(saved['performance'], {'marker': 'runtime identity'})
        self.assertNotIn('performance', config)

    def test_fixture_uses_checkpoint_rng_and_identical_real_replay_indices(self):
        rng = np.random.default_rng(42)
        saved = {'config': {'batch_size': 256}, 'replay_size': 1000, 'rng': rng.bit_generator.state}
        before = copy.deepcopy(saved)
        a, b = make_fixture(saved, 10), make_fixture(saved, 10)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[0], rng.integers(1000, size=256))
        self.assertEqual(saved, before)
        with self.assertRaisesRegex(ValueError, 'batch'):
            make_fixture(saved, 10, batch_size=128)

    def test_numerical_gate_rejects_loss_gradient_and_action_changes(self):
        reference = {'loss': .1, 'gradient': torch.tensor([1., 2., 3.]), 'actions': [1, 2, 3]}
        self.assertTrue(compare_correctness(reference, reference)['passed'])
        for field, value in (('loss', .2), ('gradient', torch.tensor([-1., -2., -3.])), ('actions', [1, 2, 4])):
            changed = {**reference, field: value}
            self.assertFalse(compare_correctness(reference, changed)['passed'])
        for field, value in (('loss', float('nan')), ('gradient', torch.tensor([1., float('inf'), 3.]))):
            self.assertFalse(compare_correctness(reference, {**reference, field: value})['passed'])

    def test_benchmark_rejects_partial_checkpoints_and_long_runs(self):
        valid = {'config': {'variant': 'map', 'batch_size': 256, 'environments': 64, 'phase': 'full',
                            'requested_episodes': 300000, 'precision': 'CUDA bfloat16'}, 'completed': 300007}
        validate_input(valid)
        for field, value in (('phase', 'smoke'), ('requested_episodes', 1000), ('variant', 'ibe_map')):
            invalid = copy.deepcopy(valid)
            invalid['config'][field] = value
            with self.assertRaises(ValueError):
                validate_input(invalid)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_input({**valid, 'completed': 70000})
        for warmup, steps in ((0, 20), (51, 20), (5, 1), (5, 201)):
            with self.assertRaises(ValueError):
                validate_limits(warmup, steps)

    def test_short_training_entry_refuses_formal_outputs_and_unbounded_jobs(self):
        root = Path(__file__).resolve().parents[1] / 'tmp' / 'performance_tests'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            args = SimpleNamespace(out=Path(temporary), stop_after_updates=0, batch_size=256)
            with self.assertRaisesRegex(ValueError, 'positive'):
                train(args, PerformanceOptions())
            args.stop_after_updates = 201
            with self.assertRaisesRegex(ValueError, 'at most 200'):
                train(args, PerformanceOptions())
            args.stop_after_updates = 100
            (args.out / 'training.json').write_text('{}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'non-performance'):
                train(args, PerformanceOptions())


if __name__ == '__main__':
    unittest.main()
