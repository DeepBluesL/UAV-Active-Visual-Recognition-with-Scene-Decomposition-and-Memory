"""Tests of end-to-end gradients, causal replay and immutable RGB ordering."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from ibe_map.model import make_model
from map_baseline.environment import LEGAL, STOP
from .images import RGBImageStore
from .model import E2EModel, E2ERunner


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, bias=False)
        self.fc = nn.Linear(8, 2048)

    def forward(self, images):
        image = self.conv2(torch.relu(self.bn1(self.conv1(images))))
        return self.fc(image.mean(dim=(2, 3)))


class LargeFeatureEncoder(nn.Module):
    """Finite feature magnitudes that cannot be represented in FP16."""
    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.linspace(-80000, 80000, 2048))

    def forward(self, images):
        return self.values[None].expand(len(images), -1)


class ArrayImages:
    def __init__(self):
        self.images = np.random.default_rng(4).integers(0, 256, (605, 8, 8, 3), dtype=np.uint8)
        self.calls = []

    def get(self, indices):
        self.calls.extend(np.asarray(indices).tolist())
        return self.images[indices].copy()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setup_runner(self, variant='ibe_map', **kwargs):
        images = ArrayImages()
        data = SimpleNamespace(boxes=np.ones((605, 4), dtype=np.float32) * 64)
        torch.manual_seed(7)
        model = E2EModel(variant, encoder=TinyEncoder(), microbatch=2, amp=False, **kwargs)
        runner = E2ERunner(data, images, 'cpu')
        history = np.array([[1, 2, 3, -1, -1, -1], [4, 5, -1, -1, -1, -1]])
        lengths = np.array([3, 2])
        return runner, model, history, lengths

    def test_same_seed_head_initialization(self):
        encoder = TinyEncoder()
        for variant in ('map', 'memo_map', 'sam_map', 'ibe_map'):
            torch.manual_seed(42)
            expected = make_model(variant)
            torch.manual_seed(42)
            actual = E2EModel(variant, encoder=encoder)
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(value, actual.head.state_dict()[name], atol=0, rtol=0)

    def test_default_feature_boundary_preserves_values_beyond_fp16_range(self):
        model = E2EModel('map', encoder=LargeFeatureEncoder(), checkpoint_encoder=False, amp=False)
        images = torch.zeros(2, 3, 4, 4)
        features = model.encode(images)
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())
        self.assertGreater(float(features.detach().abs().max()), torch.finfo(torch.float16).max)
        features.square().mean().backward()
        self.assertTrue(torch.isfinite(model.encoder.values.grad).all())
        self.assertGreater(float(model.encoder.values.grad.abs().sum()), 0.)
        model.quantize_features = True
        self.assertTrue(torch.isinf(model.encode(images)).any())

    def test_bf16_default_and_explicit_legacy_autocast_selection(self):
        model = E2EModel('map', encoder=TinyEncoder(), checkpoint_encoder=False)
        images = torch.zeros(2, 3, 8, 8)
        with patch('torch.autocast') as autocast:
            model.encode(images)
        autocast.assert_called_once_with(device_type='cpu', dtype=torch.bfloat16, enabled=False)
        legacy = E2EModel('map', encoder=TinyEncoder(), checkpoint_encoder=False,
                          encoder_precision='float16', quantize_features=True)
        with patch('torch.autocast') as autocast:
            legacy.encode(images)
        autocast.assert_called_once_with(device_type='cpu', dtype=torch.float16, enabled=False)

    def test_td_gradient_reaches_visual_layers_bn_gru_and_head(self):
        runner, model, history, lengths = self.setup_runner()
        target = copy.deepcopy(model).eval().requires_grad_(False)
        model.train()
        running_before = model.encoder.bn1.running_mean.clone()
        self.assertFalse(model.encoder.bn1.training)
        self.assertTrue(model.encoder.bn1.weight.requires_grad)
        current, following, ids = runner.transition_q(model, history, lengths)
        self.assertTrue(current.requires_grad)
        self.assertFalse(following.requires_grad)
        with torch.no_grad():
            target_next, _ = runner.next_q(target, history, lengths)
            actions = following.argmax(1)
            expected = torch.tensor([.5, -.5]) + .9 * target_next.gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(current[:, STOP], expected)
        loss.backward()
        for name in ('encoder.conv1.weight', 'encoder.conv2.weight', 'encoder.bn1.weight',
                     'encoder.fc.weight', 'head.gru.weight_ih_l0', 'head.value.2.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0, name)
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        torch.testing.assert_close(running_before, model.encoder.bn1.running_mean, atol=0, rtol=0)

    def test_current_q_reads_all_history_but_never_next_or_padding(self):
        runner, model, history, lengths = self.setup_runner()
        before = runner.current_q(model, history, lengths)
        self.assertEqual(set(runner.images.calls), {1, 2, 4})
        runner.images.images[[3, 5, 600]] = 0
        history[0, 3:] = 600
        history[1, 2:] = 600
        after = runner.current_q(model, history, lengths)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        runner.images.images[1] = 0
        changed = runner.current_q(model, history, lengths)
        self.assertGreater(float((changed[0] - before[0]).detach().abs().max()), 1e-7)

    def test_map_uses_only_current_frame_and_duplicate_ids_encode_once(self):
        runner, model, history, lengths = self.setup_runner('map')
        runner.current_q(model, history, lengths)
        self.assertEqual(runner.images.calls, [2, 4])
        runner.images.calls.clear()
        runner.prefix_q(model, np.array([[2], [2], [2]]), np.ones(3, dtype=np.int64))
        self.assertEqual(runner.images.calls, [2])

    def test_checkpointed_microbatches_preserve_gradient(self):
        runner, model, history, lengths = self.setup_runner(quantize_features=False)
        reference = copy.deepcopy(model)
        reference.microbatch = 32
        reference.checkpoint_encoder = False
        model.train(); reference.train()
        actual = runner.current_q(model, history, lengths)
        expected = runner.current_q(reference, history, lengths)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().sum().backward(); expected.square().sum().backward()
        for name, parameter in model.named_parameters():
            gradient = dict(reference.named_parameters())[name].grad
            if gradient is not None:
                torch.testing.assert_close(parameter.grad, gradient, atol=2e-7, rtol=2e-4)

    def test_target_cache_requires_frozen_model_and_invalidation(self):
        runner, model, history, lengths = self.setup_runner()
        with self.assertRaisesRegex(ValueError, 'requires_grad'):
            runner.enable_target_cache(model)
        target = copy.deepcopy(model).eval().requires_grad_(False)
        expected, _ = runner.next_q(target, history, lengths)
        runner.enable_target_cache(target)
        runner.images.calls.clear()
        first, _ = runner.next_q(target, history, lengths)
        self.assertEqual(set(runner.images.calls), {1, 2, 3, 4, 5})
        runner.images.calls.clear()
        second, _ = runner.next_q(target, history, lengths)
        self.assertEqual(runner.images.calls, [])
        torch.testing.assert_close(first, expected, atol=0, rtol=0)
        torch.testing.assert_close(second, expected, atol=0, rtol=0)
        with torch.no_grad():
            target.encoder.conv1.weight.add_(.1)
        runner.invalidate_target()
        updated, _ = runner.next_q(target, history, lengths)
        self.assertEqual(set(runner.images.calls), {1, 2, 3, 4, 5})
        self.assertGreater(float((updated - expected).abs().max()), 1e-6)

    def test_choose_legal_mask_and_no_stop(self):
        runner, model, history, lengths = self.setup_runner('map')
        model.eval()
        chosen = runner.choose(model, history, lengths, allow_stop=False)
        ids = history[np.arange(2), lengths - 1]
        self.assertTrue(LEGAL[ids, chosen].all())
        self.assertTrue((chosen != STOP).all())


class ImageStoreTests(unittest.TestCase):
    def test_rgb_cache_order_content_identity_and_invalid_sources(self):
        root = Path(__file__).resolve().parents[1] / 'tmp' / 'e2e_model_tests'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            base = Path(temporary)
            data_root = base / 'dataset'
            scene = data_root / 'multiview_imgs' / '01'
            scene.mkdir(parents=True)
            names = [f'view_{i:03d}.png' for i in range(605)]
            for i, name in enumerate(names):
                Image.new('RGB', (256, 256), (i % 256, 13, 27)).save(scene / name)
            data = SimpleNamespace(scenes=[1], names=np.asarray(names),
                observation_signature={'dataset_path': str(data_root.resolve()), 'preprocess': {}})
            images = RGBImageStore(data, data_root, base / 'unused', base / 'rgb', 'map', verbose=False)
            actual = images.get([604, 2, 300, 2])
            self.assertEqual(actual.dtype, np.uint8)
            np.testing.assert_array_equal(actual[:, 0, 0], [[92, 13, 27], [2, 13, 27], [44, 13, 27], [2, 13, 27]])
            self.assertEqual(len(images.metadata[0]['source_png_sha256']), 605)
            with self.assertRaisesRegex(ValueError, 'outside'):
                images.get([-1])
            for array in images.arrays:
                array._mmap.close()
            # A same-name content change invalidates source metadata.
            Image.new('RGB', (256, 256), (255, 1, 3)).save(scene / names[0])
            with self.assertRaisesRegex(ValueError, 'signature changed'):
                RGBImageStore(data, data_root, base / 'unused', base / 'rgb', 'map', verbose=False)


if __name__ == '__main__':
    unittest.main()
