import json

import numpy as np
import torch


def load_calibrated_wps(file: str, type='numpy') -> torch.Tensor:
    with open(file, 'r') as f:
        data = json.load(f)
    if type == 'numpy':
        wps = np.array(list(data.values()), dtype=np.float32)
    elif type == 'torch':
        wps = torch.tensor(list(data.values()), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported type: {type}")
    return wps