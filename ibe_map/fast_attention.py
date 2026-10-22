"""SAM's decomposed relative-position attention using PyTorch SDPA kernels.

The Q/K/V projections, scale and additive relative-position formula are
unchanged. Fused kernels can differ slightly in floating-point rounding.
"""
import torch
from torch.nn import functional as F
from segment_anything.modeling.image_encoder import get_rel_pos


def sam_sdpa(self, x):
    batch, height, width, _ = x.shape
    q, k, v = self.qkv(x).reshape(batch,height*width,3,self.num_heads,-1).permute(2,0,3,1,4).unbind(0)
    bias = None
    if self.use_rel_pos:
        query = q.reshape(batch*self.num_heads,height,width,-1)
        rh = torch.einsum('bhwc,hkc->bhwk',query,get_rel_pos(height,height,self.rel_pos_h))
        rw = torch.einsum('bhwc,wkc->bhwk',query,get_rel_pos(width,width,self.rel_pos_w))
        bias = (rh[...,None]+rw[:,:,:,None,:]).reshape(batch,self.num_heads,height*width,height*width)
    output = F.scaled_dot_product_attention(q,k,v,attn_mask=bias,scale=self.scale)
    return self.proj(output.transpose(1,2).reshape(batch,height,width,-1))
