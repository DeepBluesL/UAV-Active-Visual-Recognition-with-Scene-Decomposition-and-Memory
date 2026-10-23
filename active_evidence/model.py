"""Small label-free state models and deterministic causal evidence rules."""
import torch
from torch import nn

OBS_DIM = 135  # raw64, SAM64, bbox4, position3


class MLP(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,outputs))

    def forward(self, x):
        return self.net(x.float())


class Gate(MLP):
    def __init__(self):
        super().__init__(2*OBS_DIM+10,3)


class Gain(MLP):
    def __init__(self, joint=False):
        self.joint = joint
        super().__init__((OBS_DIM+5)*(2 if joint else 1),25)


def gate_input(old_obs, old_p, new_obs, new_p):
    return torch.cat((old_obs,old_p,new_obs,new_p),dim=-1)


def candidates(old_p,new_p):
    return torch.stack((new_p,old_p,(old_p+new_p)/2),dim=-2)


def gate_update(gate,old_obs,old_p,new_obs,new_p,repeated=None):
    """Only already acquired observations enter this function; no label argument.

    Return operation 0=replace, 1=retain, 2=fuse. Visibility/duplicate overrides
    ensure that missing or repeated evidence never dilutes an existing memory.
    """
    op=gate(gate_input(old_obs,old_p,new_obs,new_p)).argmax(-1)
    old_visible=old_p.sum(-1)>0
    new_visible=new_p.sum(-1)>0
    op=torch.where(~old_visible & new_visible,0,op)
    op=torch.where(~new_visible,1,op)
    if repeated is not None:
        op=torch.where(repeated,1,op)
    ps=candidates(old_p,new_p)
    os=torch.stack((new_obs,old_obs,(old_obs+new_obs)/2),dim=-2)
    rows=torch.arange(len(op),device=op.device)
    return os[rows,op],ps[rows,op],op


def state_input(obs,p,memory_obs=None,memory_p=None):
    now=torch.cat((obs,p),dim=-1)
    return now if memory_obs is None else torch.cat((now,memory_obs,memory_p),dim=-1)


def classify(p):
    return torch.where(p.sum(-1)>0,p.argmax(-1),-1)


def choose_gain(logits,legal,remaining,costs,coefficient):
    """Return STOP=24 if no affordable move has positive predicted net gain."""
    expected=logits.sigmoid()
    valid=legal[:,:24] & (costs[None,:24] <= remaining[:,None]+1e-6)
    value=expected[:,:24]-expected[:,24:25]-coefficient*costs[None,:24]
    value=value.masked_fill(~valid,-torch.inf)
    best,index=value.max(-1)
    return torch.where(best>0,index,24)
