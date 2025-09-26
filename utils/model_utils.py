import torch


def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    parms = ckpt['state_dict']
    cleaned_parms = {}
    for k, v in parms.items():
        k:str = k.replace('module.', '', 1)
        k = k.replace('model.', '', 1)
        cleaned_parms[k] = v
    model.load_state_dict(cleaned_parms, strict=False)
    return model
