"""Paper-sized memory module, with the preceding MAP heads held constant."""
from __future__ import annotations

import torch
from torch import nn

from map_baseline.model import MAPNetwork


class MemoMAPNetwork(MAPNetwork):
    """g_t=544, h_t=544, s_t=512; unidirectional, one-layer GRU.

    `forward_sequence` returns a Q vector at every time, so transition replay can
    reconstruct both current and next Q from the same complete causal history.
    Padding is only to the right; selected Q values always precede that padding.
    """
    recurrent = True

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=544, hidden_size=544, num_layers=1, batch_first=True)

    def forward_sequence(self, features, boxes, hidden=None):
        if features.ndim != 3 or boxes.shape[:2] != features.shape[:2]:
            raise ValueError('Expected [batch,time,2048] features and [batch,time,4] boxes')
        batch, steps = features.shape[:2]
        image = self.image(features.float().reshape(-1, 2048))
        box = self.box(boxes.float().reshape(-1, 4) / 256.0)
        g = torch.cat((image, box), dim=1).reshape(batch, steps, 544)
        history, hidden = self.gru(g, hidden)
        state = self.state(history.reshape(-1, 544))
        return self._action_values(state).reshape(batch, steps, 25), hidden

    def forward(self, features, boxes, hidden=None):
        """Single online observation, returning Q and the next hidden state."""
        q, hidden = self.forward_sequence(features[:, None], boxes[:, None], hidden)
        return q[:, 0], hidden


def make_model(variant):
    if variant in ('memo_map', 'ibe_map'):
        return MemoMAPNetwork()
    if variant in ('map', 'sam_map'):
        return MAPNetwork()
    raise ValueError(f'Unknown model variant: {variant}')
