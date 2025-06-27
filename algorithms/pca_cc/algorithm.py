"""
unofficial implementation of paper 
    "Illuminant estimation for color constancy: why spatial-domain methods work and the role of the color distribution"
"""

import numpy as np
from utils import *
from sklearn.decomposition import PCA

_paramters = {'n': 0.035}


def _pca(pixels: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=1)
    pca.fit(pixels)

    return pca.components_[0]


def pca_cc(img: np.ndarray, **kwargs) -> np.ndarray:
    valid_region = valid_pixels(img)
    img = img[valid_region]

    img_avg = img.mean(0)
    norm = np.linalg.norm(img_avg)
    img_avg /= norm

    projection_distance: np.ndarray = img @ img_avg
    sorted_index = projection_distance.argsort()

    num_of_pixels = int(img.shape[0] * _paramters['n'])
    indexes = np.concatenate(
        [sorted_index[:num_of_pixels], sorted_index[-num_of_pixels:]])

    selected_pixels = img[indexes]

    r_avg, g_avg, b_avg = _pca(selected_pixels)

    r_avg /= (g_avg + 1e-4)
    b_avg /= (g_avg + 1e-4)

    res = np.array([r_avg, 1., b_avg])

    return res
