import torch
import torch.nn as nn

from .fc4 import FC4
from utils import *

__all__ = ['C4', 'c4_loss']


def c4_loss(preds: list, target: torch.Tensor) -> torch.Tensor:
    weight = 1.0 / len(preds)
    accu = preds[0]
    ret = weight * angular_error_torch(accu, target)
    for i in range(1, len(preds)):
        accu = accu * preds[i]
        ret += weight * angular_error_torch(accu, target)
    return ret


class C4(nn.Module):
    """
    Unofficial implementation of C4 model from the paper "Cascading Convolutional Color Constancy"
    """

    def __init__(self, cascade_num=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cascade_num = cascade_num

        self.cascading_models = nn.ModuleList([
            FC4(confidence_weighted_pooling=False)
            for _ in range(self.cascade_num)
        ])

    def forward(self, inputs) -> torch.Tensor:
        if isinstance(inputs, dict):
            x = inputs['input']
        else:
            x = inputs
        preds = []
        for i in range(self.cascade_num):
            wp = self.cascading_models[i](x)
            preds.append(wp)
            if i != self.cascade_num - 1:
                x = self.apply_awb(x, wp)
        return preds

    def inference(self, inputs) -> torch.Tensor:
        if isinstance(inputs, dict):
            x = inputs['input']
        else:
            x = inputs
        ret = None
        for i in range(self.cascade_num):
            wp = self.cascading_models[i](x)
            if ret is None:
                ret = wp
            else:
                ret = ret * wp
            if i != self.cascade_num - 1:
                x = self.apply_awb(x, wp)
        return ret

    def apply_awb(self, x: torch.Tensor, wps: torch.Tensor) -> torch.Tensor:
        out = x / (wps[:, :, torch.newaxis, torch.newaxis] + 1e-8)
        out = out / out.max()
        return out
