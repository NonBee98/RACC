import torch
import numpy as np
from models import *
from utils import *
from params import *
import kornia

model = None

__all__ = ['pcc']


def _feature_select(img: torch.Tensor):
    """
        The four feature selected, i.e., bright, max, mean and dark pixels
        """

    feature_data = extract_statistical_features(
        img,
        channel_first=True,
        thresh_dark=PCCParams.thresh_dark,
        thresh_saturation=PCCParams.thresh_saturation)

    return feature_data


def _extract_features(img: np.ndarray) -> torch.Tensor:
    img = torch.from_numpy(img).permute(2, 0, 1)
    img_feature = _feature_select(img)
    return img_feature.unsqueeze(0)


def pcc(inputs: dict, **kwargs) -> np.ndarray:
    assert "opt" in kwargs, "opt not found in kwargs"
    img = inputs['input']
    opt = kwargs["opt"]
    if opt.ckpt is not None:
        model_path = opt.ckpt
    else:
        model_path = os.path.join(opt.checkpoint_path, opt.model_name,
                                  'best.ckpt')
    global model
    if model is None:
        model = PCC()
        load_ckpt(model, model_path)
        model.eval()
    img_feature = _extract_features(img)
    with torch.no_grad():
        pred_color = model.inference(img_feature)
    pred_color = pred_color.squeeze().numpy()
    return pred_color
