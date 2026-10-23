"""Small CPU checks for feature and evidence-gate ablations."""
import unittest

import torch

from active_evidence.ablation_model import (
    FeatureGain, FeatureGate, RuleGate, mask_embeddings,
)
from active_evidence.model import classify, gate_input, gate_update


class AblationModelTests(unittest.TestCase):
    def test_embedding_masks_are_invariant_and_preserve_other_inputs(self):
        torch.manual_seed(17)
        for width in (140, 280):
            with self.subTest(width=width):
                values = torch.randn(3, width)
                original = values.clone()
                masked = mask_embeddings(values)
                perturbed = values.clone()
                perturbed[..., :128] = torch.randn_like(perturbed[..., :128]) * 100
                if width == 280:
                    perturbed[..., 140:268] = torch.randn_like(perturbed[..., 140:268]) * 100

                torch.testing.assert_close(masked, mask_embeddings(perturbed), rtol=0, atol=0)
                torch.testing.assert_close(values, original, rtol=0, atol=0)
                # bbox/position and class-probability sections remain intact.
                torch.testing.assert_close(masked[..., 128:140], values[..., 128:140])
                if width == 280:
                    torch.testing.assert_close(masked[..., 268:280], values[..., 268:280])
                    torch.testing.assert_close(masked[..., 140:268], torch.zeros_like(masked[..., 140:268]))
                torch.testing.assert_close(masked[..., :128], torch.zeros_like(masked[..., :128]))

    def test_embedding_free_models_ignore_arbitrary_embedding_changes(self):
        torch.manual_seed(23)
        gate = FeatureGate(choices=3, embeddings=False).eval()
        gain = FeatureGain(embeddings=False).eval()
        pair = torch.randn(4, 280)
        state = torch.randn(4, 140)
        changed_pair, changed_state = pair.clone(), state.clone()
        changed_pair[..., :128] += torch.randn_like(changed_pair[..., :128]) * 1000
        changed_pair[..., 140:268] += torch.randn_like(changed_pair[..., 140:268]) * 1000
        changed_state[..., :128] += torch.randn_like(changed_state[..., :128]) * 1000
        pair_before, state_before = pair.clone(), state.clone()

        torch.testing.assert_close(gate(pair), gate(changed_pair), rtol=0, atol=0)
        torch.testing.assert_close(gain(state), gain(changed_state), rtol=0, atol=0)
        torch.testing.assert_close(pair, pair_before, rtol=0, atol=0)
        torch.testing.assert_close(state, state_before, rtol=0, atol=0)

    def test_binary_gate_has_no_fuse_head(self):
        gate = FeatureGate(choices=2).eval()
        result = gate(torch.randn(5, 280))
        self.assertEqual(tuple(result.shape), (5, 2))
        self.assertTrue(bool((result.argmax(-1) < 2).all()))

    def test_agree_rule_fuses_consensus_and_compares_confidence_on_conflict(self):
        # Four examples: consensus, stronger new, stronger old, and confidence tie.
        old_p = torch.tensor([
            [0.8, 0.2, 0, 0, 0],
            [0.8, 0.2, 0, 0, 0],
            [0.9, 0.1, 0, 0, 0],
            [0.6, 0.4, 0, 0, 0],
        ])
        new_p = torch.tensor([
            [0.7, 0.3, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0],
            [0.3, 0.7, 0, 0, 0],
            [0.4, 0.6, 0, 0, 0],
        ])
        x = torch.zeros(4, 280)
        x[:, 135:140] = old_p
        x[:, 275:280] = new_p
        self.assertEqual(RuleGate('agree')(x).argmax(-1).tolist(), [2, 0, 1, 0])
        self.assertEqual(RuleGate('latest')(x).argmax(-1).tolist(), [0, 0, 0, 0])

    def test_common_visibility_and_repeat_overrides_keep_unknown_semantics(self):
        old_obs, new_obs = torch.randn(3, 135), torch.randn(3, 135)
        old_p = torch.tensor([[0.6, 0.4, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        new_p = torch.tensor([[0, 0, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0, 0, 0, 0, 0]])
        scores = torch.zeros(3, 280)
        scores[:, 135:140] = old_p
        scores[:, 275:280] = new_p
        gate = RuleGate('latest')  # Prefer replace, then exercise shared overrides.
        _, updated, operation = gate_update(
            gate, old_obs, old_p, new_obs, new_p,
            repeated=torch.tensor([False, False, False]),
        )
        self.assertEqual(operation.tolist(), [1, 0, 1])
        self.assertEqual(classify(updated).tolist(), [0, 1, -1])

        # A repeated visible view cannot alter an existing accumulator.
        repeated_p = torch.tensor([[0.1, 0.9, 0, 0, 0]])
        _, kept, repeated_op = gate_update(
            gate, old_obs[:1], old_p[:1], new_obs[:1], repeated_p,
            repeated=torch.tensor([True]),
        )
        self.assertEqual(repeated_op.item(), 1)
        torch.testing.assert_close(kept, old_p[:1])


if __name__ == '__main__':
    unittest.main()
