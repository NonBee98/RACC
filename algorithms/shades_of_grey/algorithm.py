import numpy as np
from utils import *


def shades_of_grey(img: np.ndarray, p=6) -> np.ndarray:
    valid_reigion = valid_pixels(img)

    img = img[valid_reigion]
    img = np.power(img, p)

    illum_est = img.mean(axis=0)
    r_avg, g_avg, b_avg = np.power(illum_est, 1 / p)
    r_avg /= (g_avg + 1e-4)
    b_avg /= (g_avg + 1e-4)

    res = np.array([r_avg, 1., b_avg])

    return res
