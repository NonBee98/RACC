import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_utils import *


def linear_interpolation_ccm(M_2800K, M_6500K, cct):
    weight_k = (1 / cct - 1 / CCT_6500) / (1 / CCT_2800 - 1 / CCT_6500)
    M_xyz2rgb = weight_k * M_2800K + (1 - weight_k) * M_6500K
    return M_xyz2rgb


def calc_ang_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = torch.from_numpy(pred).float()
    gt = torch.from_numpy(gt).float()
    norm_a = torch.norm(pred)
    norm_b = torch.norm(gt)
    norm_a = pred / (norm_a + 1e-9)
    norm_b = gt / (norm_b + 1e-9)
    dot_ab = norm_a @ norm_b
    dot_ab = torch.clamp(dot_ab, -0.999999, 0.999999)
    ae = torch.rad2deg(torch.acos(dot_ab))
    ae = float(ae)
    return ae


def calc_cct_error(pred: np.ndarray, gt: np.ndarray, ccm_2800K: np.ndarray,
                   ccm_6500K: np.ndarray) -> tuple[float, float, float]:
    pred_cct, _ = cct_estimate(ccm_2800K, ccm_6500K, gt_rgb=pred)
    gt_cct, _ = cct_estimate(ccm_2800K, ccm_6500K, gt_rgb=gt)
    return abs(pred_cct - gt_cct), pred_cct, gt_cct


def angular_error_torch(pred: torch.Tensor,
                        gt: torch.Tensor,
                        average: bool = True) -> torch.Tensor:
    norm_a = torch.norm(pred, dim=-1, keepdim=True)
    norm_b = torch.norm(gt, dim=-1, keepdim=True)
    norm_a = pred / (norm_a + 1e-9)
    norm_b = gt / (norm_b + 1e-9)
    dot_ab = torch.sum(norm_a * norm_b, dim=-1)
    dot_ab = torch.clamp(dot_ab, -0.999999, 0.999999)
    ae = torch.rad2deg(torch.acos(dot_ab))
    if average:
        ae = torch.mean(ae)
    return ae


ce_loss = nn.CrossEntropyLoss(reduction='mean')


def classification_loss(pred, target):
    return ce_loss(pred, target)
