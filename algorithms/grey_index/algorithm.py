"""
unofficial implementation of paper
    "On Finding Gray Pixels"
"""

import cv2
import numpy as np

from utils import *

_params = {
    'patch_size': 5,
    'blur_kernel': 7,
    "top_n": 0.1,
    'eps': 1e-6,
    'delta_threshold': 1e-4
}


def _compute_invalid_region(img: np.ndarray) -> np.ndarray:
    mask = np.any(img > 0.95, axis=-1) | np.any(img < 0.1, axis=-1)

    return mask


def _compute_derive_gaussian(img: np.ndarray,
                             kernel_half_size=2,
                             sigma: float = .5):
    x, y = np.meshgrid(np.arange(-kernel_half_size, kernel_half_size + 1),
                       np.arange(-kernel_half_size, kernel_half_size + 1))
    ssq = sigma**2
    kernel = -x * np.exp(-(x**2 + y**2) / (2 * ssq)) / (np.pi * ssq)

    ans = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    ans = np.abs(ans)
    return ans


def _compute_grey_index_map(img: np.ndarray) -> np.ndarray:
    mask = _compute_invalid_region(img)
    img = cv2.blur(img, (7, 7), borderType=cv2.BORDER_REPLICATE)
    img[img == 0] = _params['eps']
    img = img * MAX_16BIT + 1
    norm1 = img.sum(axis=-1)
    mask = mask | np.any(img == 0, axis=-1)

    delta_r = _compute_derive_gaussian(img[..., 0])
    delta_g = _compute_derive_gaussian(img[..., 1])
    delta_b = _compute_derive_gaussian(img[..., 2])

    mask = mask | (delta_r < _params['delta_threshold']) & (
        delta_g < _params['delta_threshold']) & (delta_b <
                                                 _params['delta_threshold'])

    log_r = np.log(img[..., 0]) - np.log(norm1)
    log_b = np.log(img[..., 2]) - np.log(norm1)

    delta_log_r = _compute_derive_gaussian(log_r)
    delta_log_b = _compute_derive_gaussian(log_b)
    data = np.stack([delta_log_r, delta_log_b], axis=-1)
    mink_norm = 2
    map_uniquelight = np.linalg.norm(data, ord=mink_norm, axis=-1)
    map_uniquelight[mask] = map_uniquelight.max()

    map_uniquelight = cv2.blur(
        map_uniquelight, (_params['blur_kernel'], _params['blur_kernel']),
        borderType=cv2.BORDER_REPLICATE)

    return map_uniquelight


def grey_index(img: np.ndarray, **kwargs) -> np.ndarray:
    h, w, c = img.shape
    pixel_num = int(h * w * _params['top_n'] / 100)

    grey_index_map = _compute_grey_index_map(img)
    grey_index_map = np.ravel(grey_index_map)
    indexes = np.argsort(grey_index_map)[:pixel_num]
    img = img.reshape((-1, c))

    candidates: np.ndarray = img[indexes]
    r_avg, g_avg, b_avg = candidates.mean(axis=0)

    r_avg /= (g_avg + _params['eps'])
    b_avg /= (g_avg + _params['eps'])

    res = np.array([r_avg, 1., b_avg])

    return res
