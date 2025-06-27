import kornia
import numpy as np
import torch

from models import *
from params import *
from utils import *

model = None

__all__ = ['c5']


def _get_uv_coord(hist_size=64, tensor=True, normalize=False):
    """ Gets uv-coordinate extra channels to augment each histogram as
    mentioned in the paper.

    Args:
        hist_size: histogram dimension (scalar).
        tensor: boolean flag for input torch tensor; default is true.
        normalize: boolean flag to normalize each coordinate channel; default
        is false.

    Returns:
        u_coord: extra channel of the u coordinate values; if tensor arg is True,
        the returned tensor will be in (1 x height x width) format; otherwise,
        it will be in (height x width) format.
        v_coord: extra channel of the v coordinate values. The format is the same
        as for u_coord.
    """

    u_coord, v_coord = np.meshgrid(
        np.arange(-(hist_size - 1) / 2, ((hist_size - 1) / 2) + 1),
        np.arange((hist_size - 1) / 2, (-(hist_size - 1) / 2) - 1, -1))
    if normalize:
        u_coord = (u_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
        v_coord = (v_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
    if tensor:
        u_coord = torch.from_numpy(u_coord).to(dtype=torch.float32)
        u_coord = torch.unsqueeze(u_coord, dim=0)
        v_coord = torch.from_numpy(v_coord).to(dtype=torch.float32)
        v_coord = torch.unsqueeze(v_coord, dim=0)
    return u_coord, v_coord


def _extract_features(img: np.ndarray):
    u_coord, v_coord = _get_uv_coord(64)
    coords_map = torch.cat([u_coord, v_coord], dim=0)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    hist = compute_uv_histogram_torch(
        img,
        64,
        2,
        channel_first=True,
    )

    ret = torch.unsqueeze(hist, dim=0)

    edge_img = kornia.filters.sobel(img.unsqueeze(0)).squeeze()
    edge_hist = compute_uv_histogram_torch(edge_img, 64, 2, channel_first=True)

    edge_hist = torch.unsqueeze(edge_hist, dim=0)

    ret = torch.cat([ret, edge_hist], dim=0)
    ret = torch.cat([ret, coords_map], dim=0)
    ret = ret.unsqueeze(0)
    return ret


def c5(inputs: dict, **kwargs) -> np.ndarray:
    global model
    assert "opt" in kwargs, "opt not found in kwargs"
    opt = kwargs["opt"]
    img = inputs['input']
    if opt.ckpt is not None:
        model_path = opt.ckpt
    else:
        model_path = os.path.join(opt.checkpoint_path, opt.model_name,
                                  'best.ckpt')
    if model is None:
        model = C5()
        load_ckpt(model, model_path)
        model.eval()
    img_feature = _extract_features(img)
    with torch.no_grad():
        pred_color = model.inference(img_feature)
    pred_color = pred_color.squeeze().numpy()
    return pred_color
