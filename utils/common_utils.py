import colour
import numpy as np

CCT_2800 = 2800
CCT_6500 = 6500


def cct_estimate(m_2800k=None, m_6500k=None, eps_k=10, gt_rgb=None):
    """
    Estimate the current gt_rgb's CCT as well as its Color Correction Matrix
    based on the 2 calibration matrix in your lab

    :param m_2800k:the default xyz2rgb matrix from calibrated, whose CCT=2800k
    :param m_6500k: - CCT=6500K
    :param eps_k:   the threshold of difference CCT, can be manually changed
    :param gt_rgb:  ground truth white point OR the awb estimated rgb of raw images
    :return: The estimated CCT Matrix

    """
    assert m_2800k is not None, "Should be initial first!"

    cct_last, cct_current, cct_calculated = 2000, 3000, 4000
    step = 0

    while abs(cct_last - cct_current) > eps_k:
        if cct_current < CCT_2800:
            M_xyz2rgb = m_2800k
        elif cct_current < CCT_6500:
            " Here using a linear interpolation method based on [1/CCT] "
            weight_k = (1 / cct_current - 1 / CCT_6500) / (1 / CCT_2800 -
                                                           1 / CCT_6500)
            M_xyz2rgb = weight_k * m_2800k + (1 - weight_k) * m_6500k
        else:
            M_xyz2rgb = m_6500k

        gt_xyz = np.dot(np.linalg.inv(M_xyz2rgb),
                        gt_rgb)  # using the inverse of xyz2rgb
        gt_xyz /= gt_xyz.sum()
        gt_xyz = np.squeeze(gt_xyz)

        cct_calculated = colour.xy_to_CCT(gt_xyz[:2], method='Hernandez 1999')

        cct_last = cct_current
        cct_current = cct_calculated

        step += 1
        if step > 100:
            return cct_current, M_xyz2rgb

    return cct_current, M_xyz2rgb


def post_process_white_point_radius(wp: np.ndarray,
                                     calibrated_wps: np.ndarray,
                                     threshold=0.03) -> np.ndarray:
    dist = np.sqrt(np.sum((calibrated_wps - wp)**2, axis=1))
    index = np.argmin(dist)
    nearest_wp = calibrated_wps[index]
    nearest_dist = dist[index]
    if nearest_dist > threshold:
        target_dist = np.sqrt(threshold / nearest_dist)
        new_wp = nearest_wp + (wp - nearest_wp) * target_dist
        return new_wp
    return wp


def post_process_white_point_nearest(wp: np.ndarray,
                                     calibrated_wps: np.ndarray) -> np.ndarray:
    dist = np.sqrt(np.sum((calibrated_wps - wp)**2, axis=1))
    index = np.argmin(dist)
    nearest_wp = calibrated_wps[index]
    new_wp = wp + nearest_wp
    new_wp = new_wp / (new_wp[..., 1] + 1e-8)
    return new_wp
