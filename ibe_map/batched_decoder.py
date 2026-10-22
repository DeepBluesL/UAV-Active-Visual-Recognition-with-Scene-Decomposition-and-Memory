"""Paired-image SAM prompt decoding without a cross-product batch expansion.

The decoder calculation is adapted from Meta's Segment Anything
``segment_anything/modeling/mask_decoder.py::MaskDecoder.predict_masks``:
https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
Copyright (c) Meta Platforms, Inc. and affiliates; licensed under Apache-2.0.
The vendored upstream license is at ``third_party/segment_anything/LICENSE``.

Each prompt row must already be paired with its image embedding row. Unlike
SAM's single-image/multiple-prompt API, this helper does not repeat every image
for every prompt. Multi-mask selection happens at low resolution before the
unchanged SAM two-stage interpolation, reducing its intermediate memory use.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def decode_paired(sam, embeddings, boxes=None, points=None, multimask=False):
    """Return ``(bool masks[N,256,256], quality[N,1 or 3])`` as NumPy arrays.

    ``embeddings`` is [N,256,64,64]. Boxes are [N,4], or points are [N,2]
    (also accepting [N,1,2]); coordinates refer to the original 256px view.
    Each point is an independent positive prompt. To submit multiple points
    for one image, repeat only that image's embedding explicitly in the caller.
    If boxes and points are both supplied, each image gets both prompt types.
    """
    if embeddings.ndim != 4 or tuple(embeddings.shape[1:]) != (256, 64, 64):
        raise ValueError('Expected paired SAM embeddings with shape [N,256,64,64]')
    count = len(embeddings)
    device = embeddings.device
    if boxes is None and points is None:
        raise ValueError('Paired decoding needs at least one box or positive point per image')
    box_tensor, point_inputs = None, None
    if boxes is not None:
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
        if len(box_tensor) != count:
            raise ValueError('Box count must equal paired embedding count')
        box_tensor = box_tensor * 4
    if points is not None:
        coordinates = torch.as_tensor(points, dtype=torch.float32, device=device).reshape(-1, 1, 2)
        if len(coordinates) != count:
            raise ValueError('Point count must equal paired embedding count')
        point_inputs = (coordinates * 4, torch.ones(coordinates.shape[:2], dtype=torch.int64, device=device))
    if count == 0:
        return np.zeros((0, 256, 256), dtype=bool), np.zeros((0, 3 if multimask else 1), dtype=np.float32)

    # Image encoding may use autocast, but the established decoder path uses
    # float32. Retain that precision even if a caller has an autocast context.
    with torch.autocast(device.type, enabled=False):
        sparse, dense = sam.prompt_encoder(points=point_inputs, boxes=box_tensor, masks=None)
        decoder = sam.mask_decoder
        output_tokens = torch.cat((decoder.iou_token.weight, decoder.mask_tokens.weight), dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(count, -1, -1)
        tokens = torch.cat((output_tokens, sparse), dim=1)

        # The intentional difference from upstream predict_masks: row i is
        # already image i for prompt i. There is no repeat_interleave here.
        source = embeddings.float() + dense
        position = sam.prompt_encoder.get_dense_pe().expand(count, -1, -1, -1)
        batch, channels, height, width = source.shape
        hidden, source = decoder.transformer(source, position, tokens)
        iou_token = hidden[:, 0]
        mask_tokens = hidden[:, 1:1 + decoder.num_mask_tokens]
        source = source.transpose(1, 2).reshape(batch, channels, height, width)
        upscaled = decoder.output_upscaling(source)
        hyper = torch.stack([
            decoder.output_hypernetworks_mlps[i](mask_tokens[:, i])
            for i in range(decoder.num_mask_tokens)
        ], dim=1)
        batch, channels, height, width = upscaled.shape
        low_masks = (hyper @ upscaled.reshape(batch, channels, height * width)).reshape(batch, -1, height, width)
        quality = decoder.iou_prediction_head(iou_token)
        if multimask:
            quality = quality[:, 1:]
            candidates = low_masks[:, 1:]
            chosen = quality.argmax(1)
            low_masks = candidates[torch.arange(count, device=device), chosen][:, None]
        else:
            quality = quality[:, :1]
            low_masks = low_masks[:, :1]
        masks = sam.postprocess_masks(low_masks, (1024, 1024), (256, 256))[:, 0]
    return (masks > sam.mask_threshold).cpu().numpy(), quality.cpu().numpy()
