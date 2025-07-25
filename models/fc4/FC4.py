from typing import Union

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from .squeezenet.SqueezeNetLoader import SqueezeNetLoader
"""
FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling
* Original code: https://github.com/yuanming-hu/fc4
* Paper: https://www.microsoft.com/en-us/research/publication/fully-convolutional-color-constancy-confidence-weighted-pooling/
"""

class FC4(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1, confidence_weighted_pooling: bool = True):
        super().__init__()
        self.confidence_weighted_pooling = confidence_weighted_pooling

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])

        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True), nn.Dropout(p=0.1),
            nn.Conv2d(64,
                      4 if confidence_weighted_pooling else 3,
                      kernel_size=1,
                      stride=1), nn.ReLU(inplace=True))

    def forward(self, inputs) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        if isinstance(inputs, dict):
            x = inputs['input']
        else:
            x = inputs
        x = self.backbone(x)
        out = self.final_convs(x)

        # Confidence-weighted pooling: "out" is a set of semi-dense feature maps
        if self.confidence_weighted_pooling:
            # Per-patch color estimates (first 3 dimensions)
            rgb = normalize(out[:, :3, :, :], dim=1)

            # Confidence (last dimension)
            confidence = out[:, 3:4, :, :]

            # Confidence-weighted pooling
            pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2),
                             dim=1)

            return pred

        # Summation pooling
        pred = normalize(torch.sum(torch.sum(out, 2), 2), dim=1)

        return pred

    def inference(self, inputs) -> Tensor:
        return self.forward(inputs)
