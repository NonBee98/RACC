import kornia
import numpy as np
import torch

from models import *
from params import *
from utils import *

model_aucc = None
model_racc = None

__all__ = ['aucc', 'racc']


def _extract_features(inputs: dict) -> torch.Tensor:
    img = inputs['input']
    als = inputs['extra_input']
    img = torch.from_numpy(img).float()

    u_coord, v_coord = get_uv_coord(CustomParams.bin_num,
                                    range=CustomParams.boundary_value * 2)
    uv_map = torch.stack([u_coord, v_coord], dim=0)
    coords_map = (uv_map + CustomParams.boundary_value) / (
        2 * CustomParams.boundary_value)

    edge_img = kornia.filters.sobel(img.permute(
        2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)

    hist = compute_uv_histogram_torch(
        img,
        bin_num=CustomParams.bin_num,
        boundary_value=CustomParams.boundary_value,
        channel_first=False)
    img_feature = hist.unsqueeze(0)

    if CustomParams.edge_info:
        egde_hist = compute_uv_histogram_torch(
            edge_img,
            bin_num=CustomParams.bin_num,
            boundary_value=CustomParams.boundary_value,
            channel_first=False)
        egde_hist = egde_hist.unsqueeze(0)
        img_feature = torch.cat([img_feature, egde_hist], dim=0)

    if CustomParams.coords_map:
        img_feature = torch.cat([img_feature, coords_map], dim=0)
    img_feature = img_feature.unsqueeze(0)
    inputs['input'] = img_feature
    inputs['extra_input'] = torch.from_numpy(als).float().unsqueeze(0)
    return inputs


def aucc(inputs: dict, **kwargs) -> np.ndarray:
    global model_aucc
    assert "opt" in kwargs, "opt not found in kwargs"
    opt = kwargs["opt"]
    if opt.ckpt is not None:
        model_path = opt.ckpt
    else:
        model_path = os.path.join(opt.checkpoint_path, opt.model_name,
                                  'best.ckpt')
    if model_aucc is None:
        model_aucc = AUCC(als_channels=opt.channels)
        load_ckpt(model_aucc, model_path)
        model_aucc.eval()
    img_feature = _extract_features(inputs)
    with torch.no_grad():
        pred_color = model_aucc.inference(img_feature)
    pred_color = pred_color.squeeze().numpy()
    return pred_color


def racc(inputs: dict, **kwargs) -> np.ndarray:
    global model_racc
    assert "opt" in kwargs, "opt not found in kwargs"
    opt = kwargs["opt"]
    if opt.ckpt is not None:
        model_path = opt.ckpt
    else:
        model_path = os.path.join(opt.checkpoint_path, opt.model_name,
                                  'best.ckpt')
    if model_racc is None:
        model_racc = RACC(als_channels=opt.channels)
        load_ckpt(model_racc, model_path)
        model_racc.eval()
    img_feature = _extract_features(inputs)
    with torch.no_grad():
        pred_color = model_racc.inference(img_feature)
    pred_color = pred_color.squeeze().numpy()
    return pred_color
