"""
unofficial implementation of paper
    "Efficient Illuminant Estimation for Color Constancy Using Grey Pixels"
"""

import cv2
import numpy as np

from utils import *

_params = {'patch_size': 3, 'blur_kernel': 7, "top_n": 0.1, 'eps': 1e-6}


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


def _compute_pixel_std(img: np.ndarray) -> np.ndarray:
    mean = np.mean(img, axis=-1)
    sq_mean = np.mean(img**2, axis=-1)
    tmp = sq_mean - mean**2
    tmp[tmp < 0] = 0
    std_dev = np.sqrt(tmp)
    return std_dev


def _compute_grey_index_map(img: np.ndarray, method='std') -> np.ndarray:
    mask = np.all(img < _params['eps'], axis=-1)
    img = img * MAX_16BIT + 1
    log_img = np.log(img) + _params['eps']
    if method == 'std':
        iim_map = _compute_local_std(log_img)
    else:
        iim_map = _compute_local_derive_gaussian(log_img)
    mask |= np.all(iim_map < _params['eps'], axis=-1)

    Ds = _compute_pixel_std(iim_map)
    Ds /= (iim_map.mean(axis=-1) + _params['eps'])

    l_value = img.mean(axis=-1)

    Ps = Ds / l_value

    Ps /= (Ps.max() + _params['eps'])

    Ps[mask] = Ps.max()

    grey_index_map = cv2.blur(Ps,
                              (_params['blur_kernel'], _params['blur_kernel']))

    return grey_index_map


def grey_pixels(img: np.ndarray, **kwargs) -> np.ndarray:
    h, w, c = img.shape
    img = img.reshape(-1, c)
    pixel_num = int(h * w * _params['top_n'] / 100)

    grey_index_map = _compute_grey_index_map(img, method='std')
    grey_index_map = np.ravel(grey_index_map)
    indexes = np.argsort(grey_index_map)[:pixel_num]

    candidates: np.ndarray = img[indexes]
    r_avg, g_avg, b_avg = candidates.mean(axis=0)

    r_avg /= (g_avg + _params['eps'])
    b_avg /= (g_avg + _params['eps'])

    res = np.array([r_avg, 1., b_avg])

    return res
