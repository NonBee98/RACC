import kornia
import numpy as np
import torch

from models import *
from params import *
from utils import *

model = None

__all__ = ['als_only']


def _extract_features(inputs: dict) -> torch.Tensor:
    als = inputs['extra_input']
    inputs['input'] = None
    inputs['extra_input'] = torch.from_numpy(als).float().unsqueeze(0)
    return inputs


def als_only(inputs: dict, **kwargs) -> np.ndarray:
    global model
    assert "opt" in kwargs, "opt not found in kwargs"
    opt = kwargs["opt"]
    if opt.ckpt is not None:
        model_path = opt.ckpt
    else:
        model_path = os.path.join(opt.checkpoint_path, opt.model_name,
                                  'best.ckpt')
    if model is None:
        model = ALSOnly(als_channels=opt.channels)
        load_ckpt(model, model_path)
        model.eval()
    feature = _extract_features(inputs)
    with torch.no_grad():
        pred_color = model.inference(feature)
    pred_color = pred_color.squeeze().numpy()
    return pred_color
