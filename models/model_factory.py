from functools import partial

from dataloaders import *
from utils import *

from .c4 import *
from .c5 import *
from .custom import *
from .fc4 import *
from .onenet import *
from .pcc import *


def load_model(args, model_name: str):
    model_name = model_name.split("#")[0]
    if model_name == "pcc":
        model = PCC
        dataloader = PCCdataset
        loss_function = angular_error_torch
    elif model_name == "aucc":
        model = partial(AUCC, als_channels=args.channels)
        dataloader = RACCDataset
        loss_function = angular_error_torch
    elif model_name == "racc":
        model = partial(RACC, als_channels=args.channels)
        dataloader = RACCDataset
        loss_function = racc_loss
    elif model_name == "fc4":
        model = FC4
        dataloader = TorchDataset
        loss_function = angular_error_torch
    elif model_name == "c4":
        model = C4
        dataloader = TorchDataset
        loss_function = c4_loss
    elif model_name == "c5":
        model = C5
        dataloader = C5Dataset
        loss_function = c5_loss
    elif model_name == "onenet":
        model = OneNet
        dataloader = TorchDataset
        loss_function = angular_error_torch
    elif model_name == "als":
        model = partial(ALSOnly, als_channels=args.channels)
        dataloader = RACCDataset
        loss_function = angular_error_torch
    elif model_name == "img":
        model = ImgOnly
        dataloader = RACCDataset
        loss_function = angular_error_torch
    else:
        raise (NotImplementedError(f"Model {model_name} not implemented"))

    return model, dataloader, loss_function
