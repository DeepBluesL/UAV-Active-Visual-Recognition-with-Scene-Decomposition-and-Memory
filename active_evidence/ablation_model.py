"""Matched-capacity gate and feature ablations; no evaluator labels enter here."""
import torch

from .model import Gate, Gain


def mask_embeddings(x):
    """Remove raw/SAM embeddings, preserving bbox, position and class scores."""
    masked = x.clone()
    masked[..., :128] = 0
    if x.shape[-1] == 280:
        masked[..., 140:268] = 0
    elif x.shape[-1] != 140:
        raise ValueError('Expected a 140-dimensional state or 280-dimensional pair')
    return masked


class FeatureGate(Gate):
    def __init__(self, choices=3, embeddings=True):
        if choices not in (2, 3):
            raise ValueError('Gate choices must be 2 or 3')
        super().__init__()
        self.choices = choices
        self.embeddings = embeddings

    def forward(self, x):
        # Keep identical hidden width/initialization; the unused third head of
        # the binary variant receives no loss and can never be selected.
        return super().forward(x if self.embeddings else mask_embeddings(x))[..., :self.choices]


class FeatureGain(Gain):
    def __init__(self, embeddings=True):
        super().__init__(False)
        self.embeddings = embeddings

    def forward(self, x):
        return super().forward(x if self.embeddings else mask_embeddings(x))


class RuleGate(torch.nn.Module):
    """Fixed, untuned rules; common visibility/revisit overrides apply outside."""
    def __init__(self, name):
        super().__init__()
        if name not in ('latest', 'agree'):
            raise ValueError(name)
        self.name = name

    def forward(self, x):
        old_p, new_p = x[..., 135:140], x[..., 275:280]
        op = torch.zeros(x.shape[:-1], device=x.device, dtype=torch.long)
        if self.name == 'agree':
            op = torch.where(new_p.max(-1).values >= old_p.max(-1).values, 0, 1)
            op = torch.where(new_p.argmax(-1) == old_p.argmax(-1), 2, op)
        return torch.nn.functional.one_hot(op, 3).float()
