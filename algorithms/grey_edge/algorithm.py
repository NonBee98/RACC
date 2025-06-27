"""
Unofficial implementation of paper
    Edge-Based Color Constancy
"""

import numpy as np
from utils import *
from skimage import filters


def grey_edge(img, njet=1, mink_norm=1, sigma=1, **kwargs) -> np.ndarray:
    valid_region = valid_pixels(img)

    # pre-process img by applying gauss filter
    gauss_img = filters.gaussian(img, sigma=sigma, channel_axis=-1)

    # get njet-order derivative of the pre-processed img
    if njet == 0:
        deriv_img = [gauss_img[:, :, channel] for channel in range(3)]
    else:
        if njet == 1:
            deriv_filter = filters.sobel
        elif njet == 2:
            deriv_filter = filters.laplace
        else:
            raise ValueError("njet should be in range[0-2]! Given value is: " +
                             str(njet))
        deriv_img = [
            np.abs(deriv_filter(gauss_img[:, :, channel]))
            for channel in range(3)
        ]

    # estimate illuminations
    if mink_norm == -1:  # mink_norm = inf
        estimating_func = np.max
    else:
        estimating_func = lambda x: np.power(np.mean(np.power(x, mink_norm)), 1
                                             / mink_norm)
    r_avg, g_avg, b_avg = [estimating_func(channel[valid_region]) for channel in deriv_img]

    r_avg /= (g_avg + 1e-4)
    b_avg /= (g_avg + 1e-4)

    res = np.array([r_avg, 1., b_avg])

    return res
