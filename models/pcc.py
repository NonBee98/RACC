"""
Unofficial implementation of paper
    Color constancy from a pure color view
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from params import *
from utils import *
from .common_blocks import *

__all__ = ["PCC"]

class PCC(nn.Module):

    def __init__(self, in_feature=8, neurons=8, out_features=2, layer_num=5):
        """
        The PCC model, i.e., Simple MLP net based on 5 hidden layers, with 2 linear layer,
         i.e., the first_layer and last_layer.
        """
        super(PCC, self).__init__()
        self.first_layer = nn.Linear(in_feature, neurons)
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(neurons, neurons) for _ in range(layer_num)])
        self.last_layer = nn.Linear(neurons, out_features)
        self.init_params()

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs['input']
        else:
            x = inputs
        x = x.reshape(x.shape[0], -1)
        x = self.first_layer(x)

        for layer in self.hidden_layer:
            x = F.relu(layer(x))

        out = self.last_layer(x)
        rg = out[..., 0:1]
        bg = out[..., 1:]
        ones = torch.ones_like(rg, dtype=rg.dtype, device=rg.device)
        out = torch.cat([rg, ones, bg], dim=-1)

        return out

    def inference(self, inputs):
        return self.forward(inputs)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
