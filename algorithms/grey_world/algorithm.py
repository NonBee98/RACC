import numpy as np
from utils import *


def _compute_histogram(img: np.ndarray):
    tmp = img.sum(axis=-1)
    mask = np.abs(tmp) > 1e-3
    r_diff = img[..., 0][mask]
    g_diff = img[..., 1][mask]
    tmp = tmp[mask]
    r_xy = r_diff / tmp
    g_xy = g_diff / tmp
    mask = (0 <= r_xy) & (r_xy <= 1) & (0 <= g_xy) & (g_xy <= 1)
    r_xy: np.ndarray = r_xy[mask]
    g_xy: np.ndarray = g_xy[mask]

    bins_step = 0.01
    bin_edges_x = np.arange(0, 1.01, bins_step)
    bin_edges_y = np.arange(0, 1.01, bins_step)

    hist, r_edges, g_edges = np.histogram2d(r_xy,
                                            g_xy,
                                            bins=[bin_edges_x, bin_edges_y])
    return hist, r_edges, g_edges


def _compute_illum(img: np.ndarray):
    # compute the histogram of two difference images and add them together
    hist, r_edges, g_edges = _compute_histogram(img)

    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    g_centers = (g_edges[:-1] + g_edges[1:]) / 2

    max_count = np.max(hist)
    hist[hist < max_count * 0.1] = 0
    total_count = np.sum(hist)

    weighted_values_r = hist * r_centers[:, np.newaxis]
    weighted_values_g = hist * g_centers[np.newaxis, :]

    weighted_sum_r = np.sum(weighted_values_r)
    weighted_sum_g = np.sum(weighted_values_g)

    r_avg = weighted_sum_r / total_count
    g_avg = weighted_sum_g / total_count
    b_avg = 1 - r_avg - g_avg

    r_avg /= (g_avg + 1e-4)
    b_avg /= (g_avg + 1e-4)

    res = np.array([r_avg, 1., b_avg])

    return res


def _compute_illum_all_pixels(img: np.ndarray):
    h, w, c = img.shape
    mask = valid_pixels(img)
    img = img[mask]
    img = img.reshape(-1, c)
    r_avg, g_avg, b_avg = img.mean(axis=0)

    r_avg /= (g_avg + 1e-4)
    b_avg /= (g_avg + 1e-4)

    res = np.array([r_avg, 1., b_avg])

    return res


def grey_world(img: np.ndarray, **kwargs) -> np.ndarray:

    illum_est = _compute_illum_all_pixels(img)

    return illum_est
