"""
Unofficial implementation of paper
    Efficient-Color-Constancy-with-Local-Surface-Reflectance-Statistics
"""

import numpy as np

from utils import *


def _divide_image_into_k_patches(img: np.ndarray, patch_num: int = 10) -> list:
    h, w, _ = img.shape
    assert 1 < patch_num < min(
        h, w), "patch_num should be in range [{}, {}]".format(2, min(h, w))
    ans = []
    h_width = h // patch_num
    w_width = w // patch_num
    for i in range(patch_num):
        for j in range(patch_num):
            y_start = h_width * i
            x_start = w_width * j

            y_end = h_width * (i + 1) if i != patch_num - 1 else h
            x_end = w_width * (j + 1) if j != patch_num - 1 else w

            ans.append(img[y_start:y_end, x_start:x_end])
    return ans


def _normalize_img_by_local_max(img: np.ndarray) -> np.ndarray:
    tmp = img.reshape(-1, 3)
    maxi = np.max(tmp, axis=0, keepdims=True)
    maxi[maxi == 0] = 1
    tmp /= maxi
    ret = tmp.reshape(img.shape)
    return ret


def lsrs(img: np.ndarray, patch_num: int = 10, **kwargs) -> np.ndarray:
    img = img.copy()
    valid_region = valid_pixels(img)
    img[~valid_region] = 0.
    img_patches = _divide_image_into_k_patches(img, patch_num)
    img_patches = [_normalize_img_by_local_max(x) for x in img_patches]

    img = img.reshape(-1, 3)
    fc = np.sum(img, axis=0)
    rc = np.stack([np.sum(x.reshape(-1, 3), axis=0) for x in img_patches])
    ec = np.sum(rc, axis=0)

    r_avg, g_avg, b_avg = fc / (ec + 1e-6)

    r_avg /= (g_avg + 1e-6)
    b_avg /= (g_avg + 1e-6)

    res = np.array([r_avg, 1., b_avg])

    return res
