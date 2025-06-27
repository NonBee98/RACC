import numpy as np

from utils import *


def _find_max(img: np.ndarray):
    mask = valid_pixels(img)
    tmp = img[mask]
    max_r = tmp[..., 0].max()
    max_g = tmp[..., 1].max()
    max_b = tmp[..., 2].max()

    max_r /= (max_g + 1e-4)
    max_b /= (max_g + 1e-4)

    res = np.array([max_r, 1., max_b])
    return res


def white_patch(img: np.ndarray):
    res = _find_max(img)
    return res
