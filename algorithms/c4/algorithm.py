import torch
import numpy as np
from models import *
from utils import *
from params import *

model = None

__all__ = ['c4']


def _feature_select(img: torch.Tensor):
    return img

def _extract_features(img: np.ndarray) -> torch.Tensor:
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img_feature = _feature_select(img)
    return img_feature.unsqueeze(0)


def c4(inputs: np.ndarray, **kwargs) -> np.ndarray:
    assert "opt" in kwargs, "opt not found in kwargs"
    opt = kwargs["opt"]
    img = inputs['input']
    if opt.ckpt is not None:
        model_path = opt.ckpt
    else:
        model_path = os.path.join(opt.checkpoint_path, opt.model_name,
                                  'best.ckpt')
    global model
    if model is None:
        model = C4()
        load_ckpt(model, model_path)
        model.eval()
    img_feature = _extract_features(img)
    with torch.no_grad():
        pred_color = model.inference(img_feature)
    pred_color = pred_color.squeeze().numpy()
    return pred_color
