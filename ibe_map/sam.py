"""Actual SAM bbox/point scene decomposition, with explicit prompt choices."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/segment_anything'))
from segment_anything import build_sam_vit_b
from .fast_attention import sam_sdpa
from .batched_decoder import decode_paired

SAM_SETTINGS = {
    'model': 'official SAM ViT-B, 1024x1024 input; pretrained, NOT UEVAVD-finetuned',
    'target_prompt': 'given bbox, single-mask output; empty bbox means empty target mask',
    'surrounding_prompts': '8 independent positive points; radial contour exits +12 pixels',
    'empty_mask_prompt_fallback': 'bbox ellipse plus12 pixels; absent bbox image center with radius32',
    'object_selection': 'highest predicted IoU of3 masks for each point; no confidence filtering',
    'overlap': 'large masks painted first, small masks next, predicted target RGB restored last',
    'gray_values': 'uniform integers0..255, deterministic per image SHA256 and seed42; fixed cached view',
    'unsegmented_pixels': 'retain original RGB',
    'compute': 'encoder CUDA float16 autocast; prompt and mask decoders float32',
    'attention_backend': 'PyTorch SDPA with unchanged decomposed relative-position formula; floating-point rounding may differ',
    'decoder_batching': 'paired image/prompt rows, max32 prompts; same mask equations; minor float32 rounding differences',
}


def ring_points(mask, box, count=8, margin=12):
    """Choose outside-contour prompts using only the current predicted mask."""
    yy, xx = np.nonzero(mask)
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    if len(xx):
        cx, cy = float(xx.mean()), float(yy.mean())
    elif x2 > x1 and y2 > y1:
        cx, cy = (x1+x2)/2, (y1+y2)/2
    else:
        cx = cy = 127.5
    points = []
    for angle in np.arange(count)*2*np.pi/count:
        dx, dy = np.cos(angle), np.sin(angle)
        radii = np.arange(0, 363, dtype=float)
        xs, ys = np.rint(cx+radii*dx).astype(int), np.rint(cy+radii*dy).astype(int)
        valid = (xs>=0)&(xs<256)&(ys>=0)&(ys<256)
        xs, ys, radii = xs[valid], ys[valid], radii[valid]
        hits = np.flatnonzero(mask[ys,xs])
        if len(hits):
            radius = radii[hits[-1]] + margin
        elif x2>x1 and y2>y1:
            rx, ry = (x2-x1)/2+margin, (y2-y1)/2+margin
            radius = 1/np.sqrt((dx/rx)**2+(dy/ry)**2)
        else:
            radius = 32
        px, py = int(np.clip(round(cx+radius*dx),0,255)), int(np.clip(round(cy+radius*dy),0,255))
        if not mask[py,px] and (px,py) not in points:
            points.append((px,py))
    return np.asarray(points, dtype=np.float32).reshape(-1,2)


def compose(image, target, masks):
    """Original target RGB + random gray surrounding segments, not bbox cutouts."""
    seed = int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:8], 'little') ^ 42
    rng = np.random.default_rng(seed)
    output = image.copy()
    gray = rng.integers(0,256,len(masks),dtype=np.uint8)
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    for index in order:
        output[masks[index]] = gray[index]
    output[target] = image[target]
    return output, gray


class SAMDecomposer:
    def __init__(self, checkpoint=None, device='cuda', fast_attention=True):
        self.device = torch.device(device)
        checkpoint = checkpoint or ROOT/'output/ibe_map/weights/sam_vit_b_01ec64.pth'
        self.sam = build_sam_vit_b(str(checkpoint)).to(self.device).eval().requires_grad_(False)
        self.pool = ThreadPoolExecutor(max_workers=8)
        if fast_attention:
            for block in self.sam.image_encoder.blocks:
                block.attn.forward = types.MethodType(sam_sdpa,block.attn)

    @staticmethod
    def resize_image(image):
        return np.asarray(Image.fromarray(image).resize((1024,1024),Image.Resampling.BILINEAR))

    @torch.inference_mode()
    def encode(self, images):
        resized = np.stack(list(self.pool.map(self.resize_image,images)))
        batch = torch.from_numpy(resized).to(self.device).permute(0,3,1,2).contiguous().float()
        batch = (batch-self.sam.pixel_mean)/self.sam.pixel_std
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type=='cuda'):
            embeddings = self.sam.image_encoder(batch)
        return embeddings.float()

    @torch.inference_mode()
    def decode(self, embedding, *, boxes=None, points=None, multimask=False):
        if boxes is not None:
            boxes = torch.as_tensor(boxes,dtype=torch.float32,device=self.device).reshape(-1,4)*4
        if points is not None:
            coords = torch.as_tensor(points,dtype=torch.float32,device=self.device).reshape(-1,1,2)*4
            points = (coords, torch.ones(coords.shape[:2],dtype=torch.int64,device=self.device))
        sparse,dense = self.sam.prompt_encoder(points=points,boxes=boxes,masks=None)
        masks,quality = self.sam.mask_decoder(embedding[None],self.sam.prompt_encoder.get_dense_pe(),sparse,dense,multimask)
        masks = self.sam.postprocess_masks(masks,(1024,1024),(256,256))
        if multimask:
            masks = masks[torch.arange(len(masks),device=self.device),quality.argmax(1)]
        else:
            masks = masks[:,0]
        return (masks>self.sam.mask_threshold).cpu().numpy(), quality.cpu().numpy()

    @torch.inference_mode()
    def decompose_batch(self, images, boxes):
        embeddings = self.encode(images)
        count=len(images)
        targets=np.zeros((count,256,256),bool)
        target_qualities=np.zeros((count,1),np.float32)
        visible=[i for i,box in enumerate(boxes) if box[2]>box[0] and box[3]>box[1]]
        if visible:
            ids=torch.as_tensor(visible,device=self.device)
            masks,quality=decode_paired(self.sam,embeddings[ids],boxes=np.asarray(boxes)[visible])
            targets[visible]=masks; target_qualities[visible]=quality
        points=[ring_points(mask,box) for mask,box in zip(targets,boxes)]
        prompt_images=[i for i,p in enumerate(points) for _ in p]
        all_points=np.concatenate(points) if prompt_images else np.zeros((0,2),np.float32)
        parts,quality_parts=[],[]
        for start in range(0,len(prompt_images),32):
            ids=torch.as_tensor(prompt_images[start:start+32],device=self.device)
            masks,quality=decode_paired(self.sam,embeddings[ids],points=all_points[start:start+32],multimask=True)
            parts.append(masks); quality_parts.append(quality)
        masks=np.concatenate(parts) if parts else np.zeros((0,256,256),bool)
        qualities=np.concatenate(quality_parts) if parts else np.zeros((0,3),np.float32)
        results = []
        offset=0
        for i,image in enumerate(images):
            objects=masks[offset:offset+len(points[i])]
            quality=qualities[offset:offset+len(points[i])]
            simplified,gray=compose(image,targets[i],objects)
            results.append({'simplified':simplified,'target_mask':targets[i],'surrounding_masks':objects,
                            'points':points[i],'gray_values':gray,'target_quality':target_qualities[i:i+1],
                            'surrounding_quality':quality})
            offset+=len(points[i])
        return results

