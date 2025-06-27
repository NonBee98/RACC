"""
unofficial implementation of paper
    "Efficient Illuminant Estimation for Color Constancy Using Grey Pixels"
"""

import cv2
import numpy as np

from utils import *

_params = {'patch_size': 3, 'blur_kernel': 7, "top_n": 1, 'eps': 1e-6}


def _compute_local_std(log_img: np.ndarray) -> np.ndarray:
    mean = cv2.blur(log_img, (_params['patch_size'], _params['patch_size']),
                    borderType=cv2.BORDER_REPLICATE)
    sq_mean = cv2.blur(log_img**2,
                       (_params['patch_size'], _params['patch_size']),
                       borderType=cv2.BORDER_REPLICATE)
    tmp = sq_mean - mean**2
    tmp[tmp < 0] = 0
    std_dev = np.sqrt(tmp)
    return std_dev


def _compute_local_derive_gaussian(img: np.ndarray,
                                   kernel_half_size=2,
                                   sigma: float = .5):
    x, y = np.meshgrid(np.arange(-kernel_half_size, kernel_half_size + 1),
                       np.arange(-kernel_half_size, kernel_half_size + 1))
    ssq = sigma**2
    kernel = -x * np.exp(-(x**2 + y**2) / (2 * ssq)) / (np.pi * ssq)

    ans = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    ans = np.abs(ans)
    return ans


def _compute_grey_index_map(img: np.ndarray, method='std') -> np.ndarray:
    mask = np.all(img < _params['eps'], axis=-1)
    img = img * MAX_16BIT + 1
    log_img = np.log(img) + _params['eps']
    if method == 'std':
        iim_map = _compute_local_std(log_img)
    else:
        iim_map = _compute_local_derive_gaussian(log_img)
    mask |= np.any(iim_map < _params['eps'], axis=-1)

    iim_map_normed = np.linalg.norm(iim_map, ord=2, axis=-1, keepdims=True)
    iim_map_normed[iim_map_normed == 0] = 1.
    gt = np.ones((1,1,3), dtype=iim_map.dtype)
    gt_norm = np.linalg.norm(gt, ord=2, axis=-1, keepdims=True)

    iim_map_normed = iim_map / iim_map_normed
    gt_norm = gt / gt_norm
    dot_ab = np.sum(iim_map_normed * gt_norm, axis=-1)
    dot_ab = np.clip(dot_ab, -.99999, .99999)

    greyidx_angular = np.abs(np.arccos(dot_ab))
    grey_index_map = greyidx_angular / greyidx_angular.max()
    grey_index_map[mask] = grey_index_map.max()
    grey_index_map = cv2.blur(grey_index_map,
                              (_params['blur_kernel'], _params['blur_kernel']))

    return grey_index_map


def mean_shift_grey_pixels(img: np.ndarray, **kwargs) -> np.ndarray:
    h, w, c = img.shape
    img = img.reshape(-1, c)
    pixel_num = int(h * w * _params['top_n'] / 100)

    grey_index_map = _compute_grey_index_map(img, method='edge')
    grey_index_map = np.ravel(grey_index_map)
    indexes = np.argsort(grey_index_map)[:pixel_num]

    candidates: np.ndarray = img[indexes]
    r_avg, g_avg, b_avg = candidates.mean(axis=0)

    r_avg /= (g_avg + _params['eps'])
    b_avg /= (g_avg + _params['eps'])

    res = np.array([r_avg, 1., b_avg])

    return res
