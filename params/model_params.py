import numpy as np
import torch


class CommonParams:
    thresh_dark = 0.02
    thresh_saturation = 0.98


class CCCParams:
    bin_num = 256
    boundary_value = 2
    augmentation = True
    kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                               dtype=torch.float32)
    pyramid_level = 7
    edge_info = False


class PCCParams:
    thresh_dark = 0.02
    thresh_saturation = 0.98


class CustomParams:
    bin_num = 64
    boundary_value = 2
    edge_info = True
    augmentation = False
    coords_map = False
    color_space = 'log_uv'
