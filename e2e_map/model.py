"""End-to-end image encoders and causal replay inference.

Only BatchNorm running statistics are fixed. Convolutions, BatchNorm affine
parameters, the feature projection, GRU and Q heads all receive the TD gradient.
Keeping the statistics fixed makes an observation independent of other images
in a minibatch and prevents future replay frames leaking into current Q values.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ibe_map.model import make_model
from map_baseline.environment import LEGAL, STOP
from map_baseline.features import ResNet50, sha256
from map_baseline.model import masked_argmax


class E2EModel(nn.Module):
    def __init__(self, variant, weights_path=None, microbatch=32,
                 checkpoint_encoder=True, amp=True, encoder=None,
                 quantize_features=False, encoder_precision='bfloat16'):
        super().__init__()
        if microbatch < 1:
            raise ValueError('microbatch must be positive')
        if encoder_precision not in ('float16', 'bfloat16'):
            raise ValueError('encoder_precision must be float16 or bfloat16; use amp=False for FP32')
        # Preserve the existing policy-head initialization for a given seed.
        self.head = make_model(variant)
        self.recurrent = bool(getattr(self.head, 'recurrent', False))
        self.variant = variant
        self.microbatch = int(microbatch)
        self.checkpoint_encoder = bool(checkpoint_encoder)
        self.amp = bool(amp)
        self.encoder_precision = encoder_precision
        self.quantize_features = bool(quantize_features)
        self.pretrained_sha256 = None
        if encoder is None:
            if weights_path is None:
                raise ValueError('Official ResNet50 weights_path is required')
            self.pretrained_sha256 = sha256(weights_path)
            if not self.pretrained_sha256.startswith('0676ba61'):
                raise ValueError('Expected official ImageNet1K V1 ResNet50 weights')
            encoder = ResNet50()
            encoder.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
            # The ImageNet classifier is not part of the policy encoder.
            encoder.fc = nn.Identity()
        self.encoder = encoder
        self.encoder.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for layer in self.encoder.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def _encode_chunk(self, images):
        dtype = torch.bfloat16 if self.encoder_precision == 'bfloat16' else torch.float16
        with torch.autocast(device_type=images.device.type, dtype=dtype,
                            enabled=self.amp and images.device.type == 'cuda'):
            return self.encoder(images)

    def encode(self, images):
        """Encode normalized NCHW images without retaining stale features.

        Activation checkpointing retains the gradient through every image and
        layer; it only recomputes activations during backward. No history is
        detached. Features cross into the policy head in FP32; forcing them to
        FP16 can overflow after backbone training, even with BF16 convolutions.
        The optional legacy quantization flag exists only for numeric checks.
        """
        if images.ndim != 4 or images.shape[1] != 3 or len(images) == 0:
            raise ValueError('Expected nonempty [N,3,H,W] images')
        parts = []
        for batch in images.split(self.microbatch):
            if images.device.type == 'cuda':
                batch = batch.contiguous(memory_format=torch.channels_last)
            if self.checkpoint_encoder and torch.is_grad_enabled():
                features = checkpoint(self._encode_chunk, batch, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                features = self._encode_chunk(batch)
            if features.ndim != 2 or features.shape[1] != 2048:
                raise ValueError('Encoder must produce [N,2048] features')
            if self.quantize_features:
                features = features.to(torch.float16)
            parts.append(features.float())
        return torch.cat(parts)

    def forward(self, features, boxes, hidden=None):
        """Feature-level interface retained for fixed-checkpoint evaluation."""
        if self.recurrent:
            return self.head(features, boxes, hidden)
        return self.head(features, boxes)

    def forward_sequence(self, features, boxes, hidden=None):
        if not self.recurrent:
            raise ValueError('A nonrecurrent model does not have sequence state')
        return self.head.forward_sequence(features, boxes, hidden)


class E2ERunner:
    """Image-based counterpart of SequenceRunner with identical action masks.

    ``images.get(ids)`` returns RGB uint8 NHWC NumPy arrays. Replay stores only
    observation indices. A CNN forward is shared only for duplicate image IDs
    inside one call, which is safe because BatchNorm statistics are fixed.
    """
    def __init__(self, data, images, device='cuda', acting_batch=128):
        self.data, self.images = data, images
        self.device = torch.device(device if device != 'cuda' or torch.cuda.is_available() else 'cpu')
        self.boxes = torch.as_tensor(data.boxes, device=self.device)
        self.legal = torch.as_tensor(LEGAL, device=self.device)
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        self.acting_batch = int(acting_batch)
        self._target_model = None
        self._target_features = None
        self._target_valid = None
        self.target_cache_hits = self.target_cache_misses = 0
        if self.acting_batch < 1:
            raise ValueError('acting_batch must be positive')

    def image_features(self, model, ids):
        unique, inverse = np.unique(np.asarray(ids, dtype=np.int64), return_inverse=True)
        if model is self._target_model:
            if torch.is_grad_enabled():
                raise RuntimeError('Target feature cache is valid only under no_grad')
            missing = unique[~self._target_valid[unique]]
            self.target_cache_hits += len(unique) - len(missing)
            self.target_cache_misses += len(missing)
            if len(missing):
                features = self._encode_ids(model, missing)
                self._target_features[torch.as_tensor(missing, device=self.device)] = features
                self._target_valid[missing] = True
            return self._target_features[torch.as_tensor(unique[inverse], device=self.device)]
        encoded = self._encode_ids(model, unique)
        return encoded[torch.as_tensor(inverse, device=self.device)]

    def _encode_ids(self, model, ids):
        rgb = self.images.get(ids)
        tensor = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float()
        tensor = (tensor / 255.0 - self.mean) / self.std
        return model.encode(tensor)

    def enable_target_cache(self, target):
        """Explicit opt-in for a frozen target; invalidate at EVERY hard sync."""
        if any(parameter.requires_grad for parameter in target.parameters()):
            raise ValueError('Target model must have requires_grad_(False)')
        if target.training:
            raise ValueError('Target model must be in eval mode')
        self._target_model = target
        # FP32 avoids adding a second quantization when a test model disables it.
        self._target_features = torch.empty((len(self.boxes), 2048), device=self.device)
        self._target_valid = np.zeros(len(self.boxes), dtype=bool)

    def invalidate_target(self):
        if self._target_valid is not None:
            self._target_valid[:] = False

    def target_cache_state(self):
        """Persist valid values so resume retains the exact convolution results.

        Re-encoding the same image in a different microbatch can round slightly
        differently on CPU or CUDA. Saving this deterministic target memoization
        avoids introducing that numerical change only in interrupted runs.
        """
        if self._target_valid is None:
            return None
        ids = np.flatnonzero(self._target_valid)
        return {'ids': torch.from_numpy(ids),
                'features': self._target_features[torch.as_tensor(ids, device=self.device)].detach().cpu(),
                'hits': self.target_cache_hits, 'misses': self.target_cache_misses}

    def load_target_cache_state(self, state):
        if state is None:
            self.invalidate_target()
            return
        if self._target_valid is None:
            raise ValueError('Enable the target cache before restoring it')
        ids = state['ids'].cpu().numpy()
        features = state['features']
        if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
                or np.any(ids >= len(self.boxes)) or features.shape != (len(ids), 2048)
                or not torch.isfinite(features).all()):
            raise ValueError('Invalid saved target feature cache')
        self.invalidate_target()
        self._target_features[torch.as_tensor(ids, device=self.device)] = features.to(self.device)
        self._target_valid[ids] = True
        self.target_cache_hits, self.target_cache_misses = state['hits'], state['misses']

    @staticmethod
    def _checked_prefixes(histories, lengths):
        histories, lengths = np.asarray(histories), np.asarray(lengths, dtype=np.int64)
        if histories.ndim != 2 or lengths.shape != (len(histories),) or not len(histories):
            raise ValueError('Expected nonempty [batch,time] histories and [batch] lengths')
        if np.any(lengths < 1) or np.any(lengths > histories.shape[1]):
            raise ValueError('Invalid causal prefix length')
        valid = np.arange(histories.shape[1])[None] < lengths[:, None]
        if np.any(histories[valid] < 0):
            raise ValueError('Missing observation inside causal prefix')
        return histories, lengths

    def prefix_q(self, model, histories, lengths):
        """Q at each row's last valid observation; padding is never encoded."""
        histories, lengths = self._checked_prefixes(histories, lengths)
        rows = torch.arange(len(histories), device=self.device)
        last_ids = histories[np.arange(len(histories)), lengths - 1]
        if not model.recurrent:
            features = self.image_features(model, last_ids)
            ids = torch.as_tensor(last_ids, device=self.device)
            return model.head(features, self.boxes[ids]), ids
        longest = int(lengths.max())
        valid = np.arange(longest)[None] < lengths[:, None]
        prefix_ids = histories[:, :longest][valid]
        features = self.image_features(model, prefix_ids)
        # index_put preserves gradients into each valid CNN feature. Padding
        # contributes neither CNN work nor a selected output or loss.
        feature_sequence = torch.zeros((len(histories), longest, 2048), device=self.device)
        valid_t = torch.as_tensor(valid, device=self.device)
        feature_sequence[valid_t] = features
        box_sequence = torch.zeros((len(histories), longest, 4), device=self.device)
        box_sequence[valid_t] = self.boxes[torch.as_tensor(prefix_ids, device=self.device)]
        q, _ = model.head.forward_sequence(feature_sequence, box_sequence)
        positions = torch.as_tensor(lengths - 1, device=self.device)
        return q[rows, positions], torch.as_tensor(last_ids, device=self.device)

    def current_q(self, model, histories, lengths):
        """Replay lengths include the next observation; exclude it here."""
        return self.prefix_q(model, histories, np.asarray(lengths) - 1)[0]

    @torch.no_grad()
    def next_q(self, model, histories, lengths):
        return self.prefix_q(model, histories, lengths)

    def transition_q(self, model, histories, lengths):
        current = self.current_q(model, histories, lengths)
        following, ids = self.next_q(model, histories, lengths)
        return current, following, ids

    @torch.inference_mode()
    def choose(self, model, histories, lengths, allow_stop=True):
        chosen = []
        for start in range(0, len(histories), self.acting_batch):
            q, ids = self.prefix_q(model, histories[start:start + self.acting_batch],
                                   lengths[start:start + self.acting_batch])
            valid = self.legal[ids % 605].clone()
            if not allow_stop:
                valid[:, STOP] = False
            chosen.append(masked_argmax(q, valid).cpu().numpy())
        return np.concatenate(chosen)
