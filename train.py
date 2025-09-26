import os
import random
import warnings

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import optim, utils

import wandb
from args import *
from dataloaders import *
from models import *
from utils import *

warnings.filterwarnings("ignore")


def reproduce(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class LightningTrainer(L.LightningModule):

    def __init__(self, model, loss, args, data_transform=None, save_dir=None):
        super().__init__()
        self.model = model()
        if args.resume:
            try:
                if args.ckpt is not None:
                    load_ckpt(self.model, args.ckpt)
                else:
                    load_ckpt(self.model, os.path.join(save_dir, "best.ckpt"))
                print("Resume from ckpt")
            except:
                print("Can't find the ckpt file")
        self.loss_func = loss
        self.model = self.model.cuda()
        self.data_transform = data_transform
        self.best_ae = float("inf")
        self.validation_aes = []
        self.save_dir = save_dir
        self.best_ae_records = []
        self.best_epoch = 0
        self.args = args

    def training_step(self, batch, batch_idx):
        self.model.train()
        if self.data_transform is not None:
            batch = self.data_transform(batch)
        inputs = batch["input"]
        if "extra_input" in batch:
            extra_inputs = batch["extra_input"]
            inputs = {"input": inputs, "extra_input": extra_inputs}
        targets = batch["target"]

        out = self.model(inputs)
        loss = self.loss_func(out, targets)

        wandb.log({"Train_loss": loss})

        return loss

    def on_train_epoch_end(self):
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        inputs = batch["input"]
        illum = batch["illum"]

        if "extra_input" in batch:
            extra_inputs = batch["extra_input"]
            inputs = {"input": inputs, "extra_input": extra_inputs}

        with torch.no_grad():
            out = self.model.inference(inputs)
        try:
            ae = angular_error_torch(out, illum)
        except:
            print(out)
            exit(0)
        self.validation_aes.append(ae)

        return ae

    def on_validation_epoch_end(self):
        aes = torch.stack(self.validation_aes)
        avg_ae = aes.mean()

        wandb.log({"Val_ae": avg_ae})

        if avg_ae < self.best_ae and self.current_epoch > 3:
            self.best_ae = avg_ae
            self.best_epoch = self.current_epoch
            torch.save(aes, os.path.join(self.save_dir, "best_ae_records.pth"))
            self.trainer.save_checkpoint(os.path.join(self.save_dir, "best.ckpt"))
        self.validation_aes = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch["input"]
        y_hat = self.model(inputs)
        return y_hat

    def configure_optimizers(self):
        if self.args.optimizer == "adam":
            optimizer_class = torch.optim.Adam
            optimizer_args = {"lr": self.args.lr, "betas": (0.9, 0.999)}
        elif self.args.optimizer == "adamw":
            optimizer_class = torch.optim.AdamW
            optimizer_args = {
                "lr": self.args.lr,
                "betas": (0.9, 0.999),
                "weight_decay": 1e-2,
            }
        else:
            raise (
                NotImplementedError(
                    "Optimizer {} is not support now, please choose from {}".format(
                        self.args.optimizer, ["adam", "adamw"]
                    )
                )
            )

        model_params = self.parameters()

        self.optimizer = optimizer_class(model_params, **optimizer_args)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            total_iters=self.trainer.max_epochs,
            start_factor=1,
            end_factor=0.1,
        )
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, self.trainer.max_epochs, eta_min=1e-4)
        return [self.optimizer], [self.scheduler]


def main(args):
    torch.set_float32_matmul_precision("high")
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    reproduce(args.random_seed)

    model, dataloader, loss_function = load_model(args, model_name)
    save_dir = os.path.join(args.checkpoint_path, model_name)
    os.makedirs(save_dir, exist_ok=True)
    traniner = LightningTrainer(
        model=model, args=args, loss=loss_function, save_dir=save_dir
    )

    train_transform = Compose(
        [
            RandomCrop(size=0.7, ratio=0.8),
            RandomNoise(ratio=0.5, std=0.01),
            Resize((args.input_size, args.input_size)),
        ]
    )

    val_transform = Compose([Resize((args.input_size, args.input_size))])

    train_dataset = dataloader(
        args.data_dir,
        mode="train",
        transform=train_transform,
        cross_validation=args.cross_validation,
        fold_num=args.fold_num,
        fold_index=args.fold_index,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
    )
    valset = dataloader(
        args.val_dir,
        mode="val",
        transform=val_transform,
        cross_validation=args.cross_validation,
        fold_num=args.fold_num,
        fold_index=args.fold_index,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
    )
    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = utils.data.DataLoader(
        valset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
    )

    if args.save_iter > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename="{epoch:02d}",
            every_n_epochs=args.save_iter,
            save_top_k=-1,
        )
    else:
        checkpoint_callback = None
    trainer = L.Trainer(
        max_epochs=epochs,
        precision="32",
        accelerator="gpu",
        enable_checkpointing=True,
        gradient_clip_val=0.1,
        callbacks=checkpoint_callback,
    )

    wandb.init(
        project="awb",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": model_name,
            "dataset": "PolyuAWB",
            "batch_size": batch_size,
            "epochs": epochs,
        },
    )

    trainer.fit(
        model=traniner, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    print(
        "Best average angular error: {}, found at epoch: {}".format(
            traniner.best_ae, traniner.best_epoch
        )
    )
    print("Model saved to: {}".format(save_dir))

    wandb.finish()


if __name__ == "__main__":
    args = parse_func()
    datasets = [
        r"E:\datasets\polyu_awb\magic_6\honor_night_scenes",
        r"E:\datasets\polyu_awb\magic_6\honor_normal_scenes",
        r"E:\datasets\polyu_awb\magic_6\honor_pure_color_scenes",
    ]
    if args.cross_validation:
        model_names = ["racc", "img", "als"]
        for dataset in datasets:
            args.ori_data_dir = dataset
            args.ori_val_dir = dataset
            for model_name in model_names:
                args.model_name = model_name
                args.model_basename = model_name
                args = format_args(args)
                for i in range(args.fold_num):
                    args.fold_index = i
                    args = format_args(args)
                    print("Fold index: {}".format(i))
                    main(args)
    # for model_name in model_names:
    #     args.model_name = model_name
    #     args.model_basename = model_name
    #     args = format_args(args)
    #     if args.cross_validation:
    #         for i in range(args.fold_num):
    #             args.fold_index = i
    #             args = format_args(args)
    #             print("Fold index: {}".format(i))
    #             main(args)
    else:
        main(args)
