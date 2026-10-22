"""Explicit factorized dueling network; details unspecified by UEVAVD are documented."""
import torch
from torch import nn


class MAPNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,512),nn.ReLU())
        self.box = nn.Sequential(nn.Linear(4,32),nn.ReLU())
        self.state = nn.Sequential(nn.Linear(544,512),nn.ReLU())
        self.value = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1))
        self.direction = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,7))
        self.distance = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,4))

    def forward(self, features, boxes):
        state=self.state(torch.cat((self.image(features.float()),self.box(boxes.float()/256.0)),dim=1))
        v=self.value(state)
        direction=self.direction(state)
        distance=self.distance(state)
        direction=direction-direction.mean(1,keepdim=True)
        distance=distance-distance.mean(1,keepdim=True)
        movement=(v[:,:,None]+direction[:,:6,None]+distance[:,None,:]).flatten(1)
        stop=v+direction[:,6:7]
        return torch.cat((movement,stop),dim=1)


def masked_argmax(q, legal):
    return q.masked_fill(~legal,-torch.inf).argmax(1)
