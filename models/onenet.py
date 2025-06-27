"""
Unofficial implementation of paper
    One-net: Convolutional color constancy simplified
"""

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.nn import init

from utils import *

from .common_blocks import *

__all__ = ["OneNet"]


class OneNet(nn.Module):

    def __init__(self, in_channels=3, dropout=0.2):
        super(OneNet, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(in_channels, 64, k=1, norm=None, bias=True), nn.MaxPool2d(8, 8)
        )
        self.conv2 = nn.Sequential(
            Conv(64, 64, k=1, norm=None, bias=True), nn.MaxPool2d(8, 8)
        )
        self.conv3 = Conv(64, 128, k=1, norm=None, bias=True)
        self.conv4 = Conv(128, 64, k=1, norm=None, bias=True, dropout=dropout)
        self.conv5 = nn.Sequential(
            Conv(64, 2, k=1, norm=None, bias=True), nn.AdaptiveAvgPool2d(1)
        )

        self.init_params()

    def forward(self, inputs) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        if isinstance(inputs, dict):
            x = inputs["input"]
        else:
            x = inputs

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = x.view(x.size(0), -1)

        rg = out[..., 0:1]
        bg = out[..., 1:]
        ones = torch.ones_like(rg, dtype=rg.dtype, device=rg.device)
        out = torch.cat([rg, ones, bg], dim=-1)

        return out

    def inference(self, inputs) -> Tensor:
        return self.forward(inputs)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
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
