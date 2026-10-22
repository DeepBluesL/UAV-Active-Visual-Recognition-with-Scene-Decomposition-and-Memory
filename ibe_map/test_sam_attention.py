"""Numerical equivalence to upstream SAM attention, with nonzero position terms."""
import unittest
import torch
from .sam import SAMDecomposer  # registers the pinned upstream import directory
from .fast_attention import sam_sdpa
from segment_anything.modeling.image_encoder import Attention


class AttentionEquivalence(unittest.TestCase):
    def test_sdpa_matches_original_with_relative_positions(self):
        torch.manual_seed(174)
        for shape in ((2,3,4,32),(1,7,7,32)):
            layer = Attention(32,num_heads=4,use_rel_pos=True,input_size=shape[1:3])
            with torch.no_grad():
                layer.rel_pos_h.normal_(std=.2); layer.rel_pos_w.normal_(std=.2)
                x=torch.randn(shape)
                torch.testing.assert_close(sam_sdpa(layer,x),layer(x),atol=1e-6,rtol=1e-5)


if __name__=='__main__':
    unittest.main()
