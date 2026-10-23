"""CPU checks for causal evidence updates and physically affordable decisions."""
import unittest

import torch
from torch import nn

from active_evidence.model import (
    OBS_DIM, Gate, Gain, candidates, choose_gain, classify, gate_input,
    gate_update, state_input,
)
from map_baseline.environment import GRID, LEGAL, RANGES, STOP


class FixedGate(nn.Module):
    """Make a gate preference explicit without training a neural network."""

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, values):
        scores = values.new_zeros((len(values), 3))
        scores[:, self.operation] = 1
        return scores


def logits_for(probabilities):
    return torch.logit(torch.as_tensor(probabilities, dtype=torch.float32))


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.costs = torch.tensor(RANGES.copy(), dtype=torch.float32) * 10
        self.legal = torch.ones((1, 25), dtype=torch.bool)

    def decide(self, probabilities, remaining=40.0, coefficient=0.0, legal=None):
        return int(choose_gain(
            logits_for([probabilities]),
            self.legal if legal is None else legal,
            torch.tensor([remaining]), self.costs, coefficient,
        )[0])

    def test_zero_budget_stops_even_when_all_movements_look_better(self):
        probabilities = [0.99] * 24 + [0.01]
        self.assertEqual(self.decide(probabilities, remaining=0), STOP)

    def test_physical_boundary_masks_the_most_attractive_move(self):
        # x=+50, y=+50, z=-60 is the top corner of the stored coordinate grid.
        corner = int(((GRID == [50, 50, -60]).all(axis=1)).nonzero()[0][0])
        legal = torch.tensor(LEGAL[[corner]].copy(), dtype=torch.bool)
        self.assertFalse(bool(legal[0, 0]))  # x +10 would leave the grid.
        self.assertTrue(bool(legal[0, 4]))   # x -10 stays on the grid.
        probabilities = [0.1] * 25
        probabilities[0], probabilities[4], probabilities[24] = 0.99, 0.8, 0.4
        self.assertEqual(self.decide(probabilities, legal=legal), 4)

    def test_budget_blocks_long_move_but_allows_exactly_affordable_move(self):
        probabilities = [0.1] * 25
        probabilities[0], probabilities[3], probabilities[24] = 0.7, 0.99, 0.4
        self.assertEqual(self.decide(probabilities, remaining=10), 0)
        self.assertEqual(self.decide(probabilities, remaining=9), STOP)

    def test_no_valid_movement_stops_without_argmax_fallback(self):
        legal = torch.zeros((1, 25), dtype=torch.bool)
        legal[:, STOP] = True
        self.assertEqual(self.decide([0.99] * 24 + [0.01], legal=legal), STOP)

    def test_zero_or_negative_gain_stops(self):
        for movement_probability in (0.4, 0.5):
            with self.subTest(movement_probability=movement_probability):
                probabilities = [movement_probability] * 24 + [0.5]
                self.assertEqual(self.decide(probabilities), STOP)

    def test_cost_changes_long_move_to_short_move_and_then_stop(self):
        probabilities = [0.1] * 25
        probabilities[0], probabilities[3], probabilities[24] = 0.7, 0.9, 0.5
        self.assertEqual(self.decide(probabilities, coefficient=0), 3)
        self.assertEqual(self.decide(probabilities, coefficient=0.01), 0)
        self.assertEqual(self.decide(probabilities, coefficient=0.03), STOP)

    def test_each_episode_has_its_own_budget_and_legal_mask(self):
        probabilities = torch.full((3, 25), 0.1)
        probabilities[:, 0] = 0.9
        probabilities[:, 4] = 0.8
        probabilities[:, STOP] = 0.5
        legal = torch.ones((3, 25), dtype=torch.bool)
        legal[1, 0] = False
        chosen = choose_gain(torch.logit(probabilities), legal,
                             torch.tensor([0.0, 10.0, 10.0]), self.costs, 0.0)
        self.assertEqual(chosen.tolist(), [STOP, 4, 0])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.old_obs = torch.arange(OBS_DIM, dtype=torch.float32)[None] / OBS_DIM
        self.new_obs = -self.old_obs
        self.old_p = torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1]])
        self.new_p = torch.tensor([[0.1, 0.7, 0.1, 0.05, 0.05]])

    def update(self, operation, old_p=None, new_p=None, repeated=None):
        return gate_update(
            FixedGate(operation), self.old_obs,
            self.old_p if old_p is None else old_p,
            self.new_obs, self.new_p if new_p is None else new_p,
            repeated=repeated,
        )

    def test_replace_retain_and_fuse_update_both_evidence_and_observation(self):
        expected = [
            (self.new_obs, self.new_p),
            (self.old_obs, self.old_p),
            ((self.old_obs + self.new_obs) / 2, (self.old_p + self.new_p) / 2),
        ]
        for operation, (expected_obs, expected_p) in enumerate(expected):
            with self.subTest(operation=operation):
                obs, probability, chosen = self.update(operation)
                self.assertEqual(chosen.item(), operation)
                torch.testing.assert_close(obs, expected_obs)
                torch.testing.assert_close(probability, expected_p)
                torch.testing.assert_close(probability.sum(-1), torch.ones(1))

    def test_candidate_order_matches_replace_retain_fuse_operations(self):
        proposed = candidates(self.old_p, self.new_p)
        torch.testing.assert_close(proposed[:, 0], self.new_p)
        torch.testing.assert_close(proposed[:, 1], self.old_p)
        torch.testing.assert_close(proposed[:, 2], (self.old_p + self.new_p) / 2)

    def test_missing_new_observation_cannot_erase_existing_evidence(self):
        for preference in (0, 2):
            with self.subTest(preference=preference):
                obs, probability, chosen = self.update(
                    preference, new_p=torch.zeros_like(self.new_p))
                self.assertEqual(chosen.item(), 1)
                torch.testing.assert_close(obs, self.old_obs)
                torch.testing.assert_close(probability, self.old_p)

    def test_first_visible_observation_replaces_unknown_memory(self):
        obs, probability, chosen = self.update(1, old_p=torch.zeros_like(self.old_p))
        self.assertEqual(chosen.item(), 0)
        torch.testing.assert_close(obs, self.new_obs)
        torch.testing.assert_close(probability, self.new_p)

    def test_entirely_invisible_history_stays_unknown(self):
        _, probability, chosen = self.update(
            2, old_p=torch.zeros_like(self.old_p), new_p=torch.zeros_like(self.new_p))
        self.assertEqual(chosen.item(), 1)
        self.assertEqual(classify(probability).item(), -1)
        torch.testing.assert_close(probability, torch.zeros_like(self.old_p))

    def test_repeated_view_neither_replaces_nor_reweights_memory(self):
        for preference in (0, 2):
            with self.subTest(preference=preference):
                obs, probability, chosen = self.update(
                    preference, repeated=torch.tensor([True]))
                self.assertEqual(chosen.item(), 1)
                torch.testing.assert_close(obs, self.old_obs)
                torch.testing.assert_close(probability, self.old_p)

    def test_repeated_mask_applies_per_episode(self):
        obs, probability, chosen = gate_update(
            FixedGate(0), self.old_obs.repeat(2, 1), self.old_p.repeat(2, 1),
            self.new_obs.repeat(2, 1), self.new_p.repeat(2, 1),
            repeated=torch.tensor([True, False]),
        )
        self.assertEqual(chosen.tolist(), [1, 0])
        torch.testing.assert_close(probability, torch.cat((self.old_p, self.new_p)))
        torch.testing.assert_close(obs, torch.cat((self.old_obs, self.new_obs)))

    def test_classify_unknown_and_stable_ties(self):
        probabilities = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.6, 0.1],
        ])
        self.assertEqual(classify(probabilities).tolist(), [-1, 0, 3])


class NeuralInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)

    def test_gate_and_both_gain_models_accept_their_causal_state_shapes(self):
        obs = torch.randn(4, OBS_DIM)
        probability = torch.rand(4, 5).softmax(-1)
        old_obs = torch.randn(4, OBS_DIM)
        old_probability = torch.rand(4, 5).softmax(-1)
        checks = [
            (Gate(), gate_input(old_obs, old_probability, obs, probability), (4, 3)),
            (Gain(), state_input(obs, probability), (4, 25)),
            (Gain(joint=True), state_input(obs, probability, old_obs, old_probability), (4, 25)),
        ]
        for model, values, shape in checks:
            with self.subTest(model=type(model).__name__, inputs=values.shape[-1]):
                result = model(values.double())
                self.assertEqual(tuple(result.shape), shape)
                self.assertTrue(bool(torch.isfinite(result).all()))

    def test_unrelated_or_future_batch_rows_do_not_change_current_prediction(self):
        # A future observation can be present in a different batch row while
        # evaluating supervision, but cannot influence this row's prediction.
        for model in (Gate(), Gain(), Gain(joint=True)):
            with self.subTest(model=type(model).__name__, inputs=model.net[0].in_features):
                width = model.net[0].in_features
                current = torch.randn(1, width)
                future_a = torch.randn(3, width)
                future_b = future_a * 100 + 100
                alone = model(current)[0]
                batch_a = model(torch.cat((current, future_a)))[0]
                batch_b = model(torch.cat((current, future_b)))[0]
                torch.testing.assert_close(alone, batch_a)
                torch.testing.assert_close(alone, batch_b)

    def test_identical_observed_prefixes_give_identical_gate_and_action(self):
        gate, gain = Gate().eval(), Gain(joint=True).eval()
        observations_a = torch.randn(4, OBS_DIM)
        probabilities_a = torch.rand(4, 5).softmax(-1)
        observations_b, probabilities_b = observations_a.clone(), probabilities_a.clone()
        observations_b[2:] = -100 * observations_b[2:]
        probabilities_b[2:] = probabilities_b[2:].flip(-1)
        costs = torch.tensor(RANGES.copy(), dtype=torch.float32) * 10

        def run_acquired_prefix(observations, probabilities):
            memory_obs, memory_p = observations[:1], probabilities[:1]
            memory_obs, memory_p, operation = gate_update(
                gate, memory_obs, memory_p, observations[1:2], probabilities[1:2],
                repeated=torch.tensor([False]),
            )
            state = state_input(observations[1:2], probabilities[1:2], memory_obs, memory_p)
            logits = gain(state)
            action = choose_gain(logits, torch.ones((1, 25), dtype=torch.bool),
                                 torch.tensor([20.0]), costs, 0.001)
            return memory_obs, memory_p, operation, logits, action

        for left, right in zip(run_acquired_prefix(observations_a, probabilities_a),
                               run_acquired_prefix(observations_b, probabilities_b)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
