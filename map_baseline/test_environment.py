"""Small, cache-free unit tests for the shared MAP environment contract."""

from __future__ import annotations

import unittest

import numpy as np

from map_baseline import environment as env


def fake_dataset(*, visible: bool = True, correct: bool = True, probability: float = 0.9) -> env.Dataset:
    """Build the small subset of Dataset fields read by transition/policies."""
    data = object.__new__(env.Dataset)
    data.visible = np.full(len(env.GRID), visible, dtype=bool)
    data.correct = np.full(len(env.GRID), correct, dtype=bool)
    data.true_probability = np.full(len(env.GRID), probability, dtype=np.float32)
    data.confidence = np.full(len(env.GRID), probability, dtype=np.float32)
    return data


class SplitAndActionContractTests(unittest.TestCase):
    def test_scene_splits_are_disjoint_and_have_expected_sizes(self):
        split = env.split_manifest()
        train, simple, hard = (set(split[name]) for name in ("train", "simple", "hard"))
        self.assertEqual(len(train), 100)
        self.assertEqual(len(simple), 25)
        self.assertEqual(len(hard), 25)
        self.assertFalse(train & simple)
        self.assertFalse(train & hard)
        self.assertFalse(simple & hard)
        self.assertEqual(train | simple | hard, set(range(1, 151)))

    def test_development_train_and_validation_are_disjoint_90_10(self):
        split = env.split_manifest()
        dev_train, validation = set(split["development_train"]), set(split["development_validation"])
        self.assertEqual(len(dev_train), 90)
        self.assertEqual(len(validation), 10)
        self.assertFalse(dev_train & validation)
        self.assertEqual(dev_train | validation, set(split["train"]))

    def test_grid_and_actions(self):
        self.assertEqual(env.GRID.shape, (605, 3))
        self.assertEqual(len({tuple(row) for row in env.GRID}), 605)
        self.assertEqual(len(env.ACTION_NAMES), 25)
        self.assertEqual(len(set(env.ACTION_NAMES)), 25)
        self.assertEqual(env.STOP, 24)
        self.assertEqual(env.ACTION_NAMES[env.STOP], "STOP")

    def test_every_legal_move_is_one_axis_10_to_40m_and_stays_in_bounds(self):
        indices, actions = np.where(env.LEGAL)
        moving = actions != env.STOP
        source = env.GRID[indices[moving]].astype(int)
        target = env.GRID[env.NEXT[indices[moving], actions[moving]]].astype(int)
        delta = target - source
        nonzero_per_move = np.count_nonzero(delta, axis=1)
        self.assertTrue(np.all(nonzero_per_move == 1))
        lengths = np.abs(delta).sum(axis=1)
        self.assertTrue(np.all((lengths >= 10) & (lengths <= 40)))
        self.assertTrue(np.all(lengths % 10 == 0))
        self.assertTrue(np.all(target[:, 0:2] >= -50))
        self.assertTrue(np.all(target[:, 0:2] <= 50))
        self.assertTrue(np.all((target[:, 2] >= -100) & (target[:, 2] <= -60)))

    def test_stop_does_not_move_and_has_zero_distance_cost(self):
        indices = np.array([0, 274, 604])
        data = fake_dataset()
        following, reward, done, success = data.transition(indices, np.full(3, env.STOP), 0)
        np.testing.assert_array_equal(following, indices)
        np.testing.assert_array_equal(done, np.ones(3, dtype=bool))
        np.testing.assert_array_equal(success, np.ones(3, dtype=bool))
        np.testing.assert_allclose(reward, [0.5, 0.5, 0.5])


class RewardAndHorizonTests(unittest.TestCase):
    def test_nonterminal_reward_includes_distance_cost(self):
        data = fake_dataset()
        idx = 274
        action = 0  # +10m on y
        _, reward, done, _ = data.transition([idx], [action], 0)
        self.assertFalse(bool(done[0]))
        self.assertAlmostEqual(float(reward[0]), -0.1 - 0.02 * 1)

    def test_terminal_correct_and_incorrect_rewards_include_distance_cost(self):
        idx, action = 274, 1  # +20m on y
        correct = fake_dataset(visible=True, correct=True, probability=0.9)
        _, reward_ok, done_ok, success_ok = correct.transition([idx], [action], 4)
        self.assertTrue(bool(done_ok[0]))
        self.assertTrue(bool(success_ok[0]))
        self.assertAlmostEqual(float(reward_ok[0]), 0.5 - 0.02 * 2)

        incorrect = fake_dataset(visible=True, correct=False, probability=0.1)
        _, reward_bad, done_bad, success_bad = incorrect.transition([idx], [action], 4)
        self.assertTrue(bool(done_bad[0]))
        self.assertFalse(bool(success_bad[0]))
        self.assertAlmostEqual(float(reward_bad[0]), -0.5 - 0.02 * 2)

    def test_correct_stop_gets_terminal_reward_with_zero_travel_cost(self):
        data = fake_dataset()
        _, reward, done, success = data.transition([274], [env.STOP], 0)
        self.assertTrue(bool(done[0]))
        self.assertTrue(bool(success[0]))
        self.assertAlmostEqual(float(reward[0]), 0.5)

    def test_fifth_decision_can_move_then_terminate(self):
        data = fake_dataset()
        idx, action = 274, 0
        following, reward, done, success = data.transition([idx], [action], 4)
        self.assertNotEqual(int(following[0]), idx)
        self.assertTrue(bool(done[0]))
        self.assertTrue(bool(success[0]))
        self.assertAlmostEqual(float(reward[0]), 0.5 - 0.02)

    def test_invisible_empty_box_cannot_be_rewarded_as_correct(self):
        data = fake_dataset(visible=False, correct=True, probability=0.99)
        _, reward, done, success = data.transition([274], [env.STOP], 0)
        self.assertTrue(bool(done[0]))
        self.assertFalse(bool(success[0]))
        self.assertAlmostEqual(float(reward[0]), -0.5)

    def test_success_mode_distinguishes_true_probability_from_top1(self):
        # The target class probability exceeds threshold, but the top prediction is wrong.
        data = fake_dataset(visible=True, correct=False, probability=0.3)
        prob_only = env.Protocol(success_mode="true_probability")
        top1_and_prob = env.Protocol(success_mode="top1_and_true_probability")
        _, _, _, success_prob = data.transition([274], [env.STOP], 0, prob_only)
        _, _, _, success_top1 = data.transition([274], [env.STOP], 0, top1_and_prob)
        self.assertTrue(bool(success_prob[0]))
        self.assertFalse(bool(success_top1[0]))


class HeuristicPolicyTests(unittest.TestCase):
    def test_fixed_three_moves_uses_same_budget_and_respects_legal_mask(self):
        data = fake_dataset()
        index = 274
        for decision in range(3):
            action = int(env.heuristic_actions(
                data, np.array([index]), decision, "heuristic", np.random.default_rng(7),
                stop_mode="fixed_moves", fixed_moves=3,
            )[0])
            self.assertNotEqual(action, env.STOP)
            self.assertTrue(env.LEGAL[index, action])
            index = int(env.NEXT[index, action])
        stop_action = int(env.heuristic_actions(
            data, np.array([index]), 3, "heuristic", np.random.default_rng(7),
            stop_mode="fixed_moves", fixed_moves=3,
        )[0])
        self.assertEqual(stop_action, env.STOP)
        self.assertTrue(env.LEGAL[index, stop_action])


if __name__ == "__main__":
    unittest.main()
