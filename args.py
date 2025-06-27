import argparse
import os
import re


def parse_func():
    parse = argparse.ArgumentParser(description="Training Options")
    parse.add_argument(
        "--project_name",
        type=str,
        default="UCC + ALS based AWB for dominant color scene",
    )

    parse.add_argument(
        "--version",
        type=str,
        default="0.0.1",
        help="the version of current dataset and model \
                           should be formated as 'version.release.modifications'",
    )

    parse.add_argument(
        "--model_name", type=str, default="racc", help="the name of the model"
    )

    parse.add_argument(
        "--channels", type=int, default=9, help="number of spectrum channels"
    )

    parse.add_argument(
        "--data_dir", type=str, default=r"", help="directory storing the training data"
    )

    parse.add_argument(
        "--val_dir", type=str, default=r"", help="directory storing the validation data"
    )

    parse.add_argument(
        "--test_dir",
        type=str,
        default=r"./input",
        help="directory storing the test data",
    )

    parse.add_argument("--out_dir", type=str, default="output", help="output directory")

    parse.add_argument(
        "--epochs", type=int, default=500, help="number of total training epochs"
    )

    parse.add_argument(
        "--cross_validation",
        type=int,
        default=0,
        help="whether to use cross validation, when it is true, \
              fold_num models will be trained and evaluated independently",
    )

    parse.add_argument(
        "--fold_num", type=int, default=5, help="the number of fold of cross validation"
    )

    parse.add_argument("--fold_index", type=int, default=0, help="the index of fold")

    parse.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/best.ckpt",
        help="path to checkpoint",
    )

    parse.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size (in distributed mode, it represents data fed to each GPU)",
    )

    parse.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="the name of optimizer, currently support ['adam', 'adamw']",
    )

    parse.add_argument("--lr", type=float, default=2e-3, help="learning rate")

    parse.add_argument("--input_size", type=int, default=256, help="input image size")

    parse.add_argument(
        "--random_seed", type=int, default=0, help="the random seed in training"
    )

    parse.add_argument(
        "--resume", type=int, default=0, help="whether to resume training"
    )

    parse.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints",
    )

    parse.add_argument(
        "--num_workers", type=int, default=12, help="number of workers in dataloader"
    )

    parse.add_argument(
        "--save_iter",
        type=int,
        default=100,
        help="save model checkpoint after certain epochs",
    )

    parse.add_argument(
        "--shuffle",
        type=int,
        default=1,
        help="whether to shuffle data, results should be different only when cross validation is used",
    )

    parse.add_argument(
        "--post_process",
        type=int,
        default=0,
        help="whether to post process the predicted white point",
    )

    parse.add_argument("--tag", type=str, default="", help="tag for the model")

    opt = parse.parse_args()
    verify_args(opt)
    batch_ratio = opt.batch_size / 32
    opt.model_basename = opt.model_name
    opt.lr = opt.lr * batch_ratio

    opt = format_args(opt)
    return opt


def format_args(opt):
    opt.data_dir = opt.data_dir.split(",")
    opt.data_dir = [x.strip() for x in opt.data_dir]
    opt.dataset_name = list(map(os.path.basename, opt.data_dir))
    opt.dataset_name = "#".join(opt.dataset_name)

    opt.val_dir = opt.val_dir.split(",")
    opt.val_dir = [x.strip() for x in opt.val_dir]
    opt.val_dataset_name = list(map(os.path.basename, opt.val_dir))
    opt.val_dataset_name = "#".join(opt.val_dataset_name)
    opt.dataset_name = "{}&{}".format(opt.dataset_name, opt.val_dataset_name)

    opt.test_dir = opt.test_dir.split(",")
    opt.test_dir = [x.strip() for x in opt.test_dir]
    opt.test_dataset_name = list(map(os.path.basename, opt.test_dir))
    opt.test_dataset_name = "#".join(opt.test_dataset_name)

    if opt.cross_validation != 0:
        assert opt.fold_num > 1, "fold_num should be greater than 1"
        assert (
            opt.fold_index >= 0 and opt.fold_index < opt.fold_num
        ), "fold_index should be in [0, fold_num)"
        if opt.shuffle == 0:
            opt.tag = "fold_{}_{}".format(opt.fold_index, opt.fold_num)
        else:
            opt.tag = "fold_{}_{}_shuffle".format(opt.fold_index, opt.fold_num)
    if opt.tag != "" and opt.tag is not None:
        opt.model_tagged_name = "{}#{}".format(opt.model_basename, opt.tag)
    else:
        opt.model_tagged_name = opt.model_basename
    opt.model_name = "{}#{}".format(opt.model_tagged_name, opt.dataset_name)
    return opt


def verify_args(opt):
    pass
