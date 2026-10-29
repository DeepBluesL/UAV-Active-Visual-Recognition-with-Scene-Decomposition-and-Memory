"""Causality, history reconstruction and matched-baseline regression checks."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from map_baseline.environment import Protocol, GRID, Dataset
from map_baseline.experiment import Runner as OldRunner, rollout as old_rollout
from map_baseline.model import MAPNetwork
from .experiment import (PrefixReplay, SequenceRunner, rollout, td_loss, verify_shared_observations,
                         validate_cache_for_variant, train, evaluate)
from .model import MemoMAPNetwork, make_model


class RecurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.model = MemoMAPNetwork().eval()
        self.features = torch.randn(3, 6, 2048)
        self.boxes = torch.rand(3, 6, 4) * 256

    def test_paper_dimensions_and_outputs(self):
        q, hidden = self.model.forward_sequence(self.features, self.boxes)
        self.assertEqual(q.shape, (3, 6, 25))
        self.assertEqual(hidden.shape, (1, 3, 544))
        self.assertEqual(self.model.gru.input_size, 544)
        self.assertEqual(self.model.state[0].out_features, 512)

    def test_carried_hidden_matches_complete_prefix(self):
        with torch.no_grad():
            q, hidden = self.model.forward_sequence(self.features, self.boxes)
            state, sequence = None, []
            for t in range(6):
                step, state = self.model(self.features[:, t], self.boxes[:, t], state)
                sequence.append(step)
        torch.testing.assert_close(torch.stack(sequence, 1), q, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(state, hidden, atol=2e-6, rtol=2e-5)

    def test_future_observation_cannot_affect_current_action_values(self):
        altered = self.features.clone()
        altered[:, 3:] = torch.randn_like(altered[:, 3:]) * 100
        with torch.no_grad():
            a, _ = self.model.forward_sequence(self.features, self.boxes)
            b, _ = self.model.forward_sequence(altered, self.boxes)
        torch.testing.assert_close(a[:, :3], b[:, :3], atol=0, rtol=0)

    def test_reset_removes_previous_episode_memory(self):
        with torch.no_grad():
            _, previous = self.model.forward_sequence(self.features[:, :4], self.boxes[:, :4])
            reset, _ = self.model(self.features[:, 4], self.boxes[:, 4], None)
            isolated, _ = self.model.forward_sequence(self.features[:, 4:5], self.boxes[:, 4:5])
            carried, _ = self.model(self.features[:, 4], self.boxes[:, 4], previous)
        torch.testing.assert_close(reset, isolated[:, 0])
        self.assertGreater(float((carried - reset).abs().max()), 1e-5)

    def test_same_current_view_can_depend_on_history(self):
        alternative = self.features.clone()
        alternative[:, :2] *= -3
        with torch.no_grad():
            a, _ = self.model.forward_sequence(self.features[:, :3], self.boxes[:, :3])
            b, _ = self.model.forward_sequence(alternative[:, :3], self.boxes[:, :3])
        self.assertGreater(float((a[:, 2] - b[:, 2]).abs().max()), 1e-5)

    def test_replay_copies_full_prefix_before_reset(self):
        replay = PrefixReplay(4)
        history = np.array([[5, 7, 8, -1, -1, -1], [100, 101, -1, -1, -1, -1]])
        replay.add(history, np.array([3, 2]), np.array([1, 24]), np.array([-.12, .5]), np.array([False, True]))
        history[:] = -1
        np.testing.assert_array_equal(replay.histories[0], [5, 7, 8, -1, -1, -1])
        self.assertEqual(replay.lengths[1], 2)
        self.assertTrue(replay.done[1])
        new = np.array([[200, 201, -1, -1, -1, -1]] * 3)
        replay.add(new, np.array([2, 2, 2]), np.array([0, 0, 0]), np.zeros(3), np.zeros(3, bool))
        self.assertEqual(replay.size, 4)
        self.assertEqual(replay.pos, 1)
        self.assertEqual(replay.histories[0, 0], 200)
        self.assertEqual(replay.histories[1, 0], 100)

    def _runner_and_replay(self):
        rng = np.random.default_rng(1)
        data = SimpleNamespace(features=rng.normal(size=(605, 2048)).astype(np.float32),
                               boxes=rng.uniform(0, 256, size=(605, 4)).astype(np.float32))
        runner = SequenceRunner(data, 'cpu')
        replay = PrefixReplay(4)
        history = np.array([[1, 2, -1, -1, -1, -1], [3, 4, 5, 6, -1, -1]])
        replay.add(history, np.array([2, 4]), np.array([24, 24]), np.array([.5, -.5]), np.array([True, True]))
        return runner, replay

    def test_padding_does_not_enter_loss_or_gradients(self):
        runner, replay = self._runner_and_replay()
        target = copy.deepcopy(self.model)
        sample = np.array([0, 1])
        before = td_loss(runner, self.model, target, replay, sample, .9)
        replay.histories[0, 2:] = 500
        replay.histories[1, 4:] = 501
        after = td_loss(runner, self.model, target, replay, sample, .9)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        runner.features.requires_grad_(True)
        td_loss(runner, self.model, target, replay, sample, .9).backward()
        self.assertEqual(float(runner.features.grad[500:502].abs().sum()), 0.)
        # The next view is used only by detached target construction, never by
        # the current estimate or its gradient.
        self.assertEqual(float(runner.features.grad[[2, 6]].abs().sum()), 0.)
        self.assertGreater(float(runner.features.grad[[1, 3, 4, 5]].abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_terminal_transition_does_not_bootstrap(self):
        runner, replay = self._runner_and_replay()
        target = copy.deepcopy(self.model)
        before = td_loss(runner, self.model, target, replay, np.array([0, 1]), .9)
        with torch.no_grad():
            for p in target.parameters():
                p.add_(10)
        after = td_loss(runner, self.model, target, replay, np.array([0, 1]), .9)
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_transition_q_uses_entire_current_and_next_prefix(self):
        runner, replay = self._runner_and_replay()
        with torch.no_grad():
            current, following, ids = runner.transition_q(self.model, replay.histories[:2], replay.lengths[:2])
            for row in range(2):
                indices = replay.histories[row, :replay.lengths[row]]
                expected, _ = self.model.forward_sequence(runner.features[indices][None], runner.boxes[indices][None])
                torch.testing.assert_close(current[row], expected[0, -2], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(following[row], expected[0, -1], atol=2e-6, rtol=2e-5)
                self.assertEqual(ids[row], indices[-1])

    def test_sam_only_architecture_matches_original_map(self):
        self.assertIsInstance(make_model('sam_map'), MAPNetwork)
        self.assertFalse(make_model('sam_map').recurrent)
        self.assertEqual(list(make_model('map').state_dict()), list(make_model('sam_map').state_dict()))

    def test_cache_comparison_rejects_classifier_or_order_changes(self):
        raw = SimpleNamespace(scenes=[1], observation_signature={
            'classifier_sha256': 'classifier', 'dataset_path': 'dataset', 'pretrained_sha256': 'encoder'},
            boxes=np.zeros((2, 4)), probs=np.zeros((2, 5)), visible=np.array([True, False]), names=np.array(['a', 'b']))
        sam = copy.deepcopy(raw)
        verify_shared_observations(raw, sam)
        sam.probs[0, 0] = .3
        with self.assertRaisesRegex(ValueError, 'probs'):
            verify_shared_observations(raw, sam)
        sam = copy.deepcopy(raw)
        sam.observation_signature['classifier_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'classifier_sha256'):
            verify_shared_observations(raw, sam)

    def test_ablation_cache_type_cannot_be_mislabeled(self):
        raw = SimpleNamespace(observation_signature={'preprocess': {'encoder': 'ResNet50'}})
        sam = copy.deepcopy(raw)
        sam.observation_signature['preprocess']['scene_decomposition'] = {'model': 'official SAM ViT-B'}
        for variant in ('stationary', 'sweep', 'random', 'map', 'memo_map'):
            validate_cache_for_variant(raw, variant)
            with self.assertRaisesRegex(ValueError, 'raw-image'):
                validate_cache_for_variant(sam, variant)
        for variant in ('sam_map', 'ibe_map'):
            validate_cache_for_variant(sam, variant)
            with self.assertRaisesRegex(ValueError, 'requires a SAM'):
                validate_cache_for_variant(raw, variant)
        for invalid in (None, {}, 'SAM'):
            sam.observation_signature['preprocess']['scene_decomposition'] = invalid
            with self.assertRaisesRegex(ValueError, 'requires a SAM'):
                validate_cache_for_variant(sam, 'ibe_map')

    def test_train_rejects_wrong_cache_before_constructing_runner(self):
        raw = SimpleNamespace(observation_signature={'preprocess': {}})
        args = SimpleNamespace(seed=42, success_mode='top1_and_true_probability', scenes=[1],
                               phase='full', cache=Path('unused'), variant='ibe_map')
        with patch('ibe_map.experiment.Dataset', return_value=raw), \
                patch('ibe_map.experiment.SequenceRunner') as runner, \
                patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'requires a SAM'):
                train(args)
            runner.assert_not_called()

    def test_evaluate_rejects_wrong_raw_and_sam_cache_before_checkpoint_load(self):
        raw = SimpleNamespace(observation_signature={'preprocess': {}})
        sam = SimpleNamespace(observation_signature={'preprocess': {'scene_decomposition': {'model': 'SAM'}}})
        args = SimpleNamespace(splits=['hard'], raw_cache=Path('raw'), sam_cache=Path('sam'), policies=['ibe_map'])
        with patch('ibe_map.experiment.Dataset', return_value=sam), patch('torch.load') as load:
            with self.assertRaisesRegex(ValueError, 'raw-image'):
                evaluate(args)
            load.assert_not_called()
        with patch('ibe_map.experiment.Dataset', side_effect=[raw, raw]), patch('torch.load') as load:
            with self.assertRaisesRegex(ValueError, 'requires a SAM'):
                evaluate(args)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
