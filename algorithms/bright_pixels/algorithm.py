"""
Unoffical implementation of paper
    MaxRGB Reconsidered
"""

from utils import *
import numpy as np


def bright_pixels(img, top_n=2, **kwargs) -> np.ndarray:
    valid_region = valid_pixels(img)
    valid_img = img[valid_region]
    luminance = valid_img.sum(axis=-1)
    indexes = luminance.argsort()[::-1]
    bright_pixels_num = int(len(valid_img) * top_n / 100)
    indexes = indexes[:bright_pixels_num]
    bright_pixels_list = valid_img[indexes]

    r_avg, g_avg, b_avg = bright_pixels_list.mean(axis=0)

    r_avg /= (g_avg + 1e-4)
    b_avg /= (g_avg + 1e-4)

    res = np.array([r_avg, 1., b_avg])

    return res
