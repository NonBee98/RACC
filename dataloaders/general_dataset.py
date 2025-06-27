import glob
import json
import os

import cv2
import numpy as np
import random

import torchvision.transforms.functional as TF
import torch

from utils import *


def read_img(path) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if img.max() > 255:
        img /= 65535.0
    else:
        img /= 255.0
    img = np.clip(img, 0, 1)
    img = img.astype(np.float16)
    return img


def read_dng_img(path, resoulution=None) -> np.ndarray:
    exif = extract_exif(path)
    demosaiced_img = preprocess_raw(path, exif, resoulution=resoulution)
    return demosaiced_img


def _get_keys(data_dict: dict, keys: list):
    for k in keys:
        if k in data_dict:
            return data_dict[k]
    return None


class GeneralDataset:

    def __init__(self,
                 root_dir,
                 extension='png',
                 mode='all',
                 img_size=(-1, -1),
                 cross_validation=False,
                 fold_num=3,
                 fold_index=0,
                 random_seed=0,
                 shuffle=False):
        self.img_size = img_size
        self.mode = mode

        assert mode in ['train', 'val', 'test',
                        'all'], "Invalid mode: {}".format(mode)
        assert fold_index < fold_num, "Invalid fold index: {}".format(
            fold_index)
        if not isinstance(root_dir, (list, tuple)):
            root_dir = [root_dir]
        self.mode = mode
        self.data_list = []
        self.gt_list = []
        file_list = []
        self.all_wps = []
        for dir in root_dir:
            data_list_file = os.path.join(dir, 'data_list.json')
            with open(data_list_file, 'r') as f:
                data_list = json.load(f)
            tmp_file_list = []
            if mode == 'all' or cross_validation:
                if 'all' in data_list:
                    file_list.extend(data_list['all'])
                    tmp_file_list.extend(data_list['all'])
                else:
                    for k in data_list.keys():
                        file_list.extend(data_list[k])
                        tmp_file_list.extend(data_list[k])
            else:
                file_list.extend(data_list[mode])
                tmp_file_list.extend(data_list[mode])

            img_dir = os.path.join(dir, 'demosaiced')
            gt_dir = os.path.join(dir, 'gt')

            for file in tmp_file_list:
                file = str(file)
                raw_file = os.path.join(img_dir, file + '.' + extension)
                gt_file = os.path.join(gt_dir, file + '.json')
                if os.path.exists(raw_file) and os.path.exists(gt_file):
                    self.data_list.append(raw_file)
                    self.gt_list.append(gt_file)
                    with open(gt_file, 'r') as f:
                        data = json.load(f)
                        self.all_wps.append(data['white_point'])
                else:
                    print("can't find file: {}, {}".format(raw_file, gt_file))

        self.scene_name = file_list
        if shuffle:
            random.seed(random_seed)
            indexes = list(range(len(self.data_list)))
            random.shuffle(indexes)
            self.data_list = [self.data_list[i] for i in indexes]
            self.gt_list = [self.gt_list[i] for i in indexes]
            self.scene_name = [self.scene_name[i] for i in indexes]

        if cross_validation:
            assert mode in [
                'train', 'val', 'test'
            ], "Invalid mode for cross validation: {}".format(mode)
            assert fold_num > 1, "fold_num should be greater than 1"
            self.data_list = self.split_fold(self.data_list, fold_num,
                                             fold_index)
            self.gt_list = self.split_fold(self.gt_list, fold_num, fold_index)
            self.scene_name = self.split_fold(self.scene_name, fold_num,
                                              fold_index)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        gt_path = self.gt_list[index]
        img = read_img(img_path)
        size = self.img_size
        if size[0] > 1 and size[1] > 1:
            input_img = torch.from_numpy(img).permute(2, 0, 1).float()
            input_img = TF.resize(input_img,
                                  size,
                                  interpolation=TF.InterpolationMode.BICUBIC)
            input_img = torch.clamp(input_img, 0, 1)
            input_img = input_img.permute(1, 2, 0).numpy()
        else:
            input_img = img

        with open(gt_path, 'r') as f:
            gt_dict = json.load(f)
        illum = np.array(gt_dict['white_point'], dtype=np.float32)
        als = np.array(gt_dict['als'], dtype=np.float32)
        als = als / (als.max() + 1e-8)
        scene_name = self.scene_name[index]

        ret = {
            'img': img,
            'input': input_img.astype(np.float32),
            'als': als,
            'illum': illum,
            'scene_name': scene_name,
        }
        ret['cm_65'] = np.array(_get_keys(gt_dict, ['cm_65', 'ColorMatrix1']))
        ret['cm_28'] = np.array(_get_keys(gt_dict, ['cm_28', 'ColorMatrix2']))
        return ret

    def split_fold(self, data_list, fold_num, fold_index):
        folds = []
        fold_size = len(data_list) // fold_num
        for i in range(fold_num):
            start_index = i * fold_size
            end_index = (i + 1) * fold_size
            if i == fold_num - 1:
                end_index = len(data_list)
            folds.append(data_list[start_index:end_index])
        if self.mode != 'train':
            return folds[fold_index]

        remain_index = [i for i in range(fold_num) if i != fold_index]
        ret = []
        for i in remain_index:
            ret += folds[i]
        return ret


class TestDataset:

    def __init__(self, root_dir, img_extension='png', img_size=(-1, -1)):
        self.txt_list = []
        if not isinstance(root_dir, (list, tuple)):
            root_dir = [root_dir]
        for dir in root_dir:
            txt_list = glob.glob(os.path.join(dir, '*.txt'))
            self.txt_list.extend(txt_list)
        self.img_size = img_size
        self.img_extension = img_extension

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, index):
        txt_path = self.txt_list[index]
        scene_name = os.path.basename(txt_path).split('.')[0]
        with open(txt_path, 'r') as f:
            lines = f.readline().strip().split()
        als = np.array(list(map(float, lines)), dtype=np.float32)
        als = als / (als.max() + 1e-8)
        img_path = txt_path.replace('.txt', '.' + self.img_extension)

        if self.img_extension == 'dng':
            img = read_dng_img(img_path)
        elif self.img_extension == 'png':
            img = read_img(img_path)
        else:
            raise NotImplementedError("Unsupported image extension: {}".format(
                self.img_extension))

        size = self.img_size
        if size[0] > 1 and size[1] > 1:
            input_img = torch.from_numpy(img).permute(2, 0, 1).float()
            input_img = TF.resize(input_img,
                                  size,
                                  interpolation=TF.InterpolationMode.BICUBIC)
            input_img = torch.clamp(input_img, 0, 1)
            input_img = input_img.permute(1, 2, 0).numpy()
        else:
            input_img = img

        ret = {
            'img': img,
            'input': input_img,
            'als': als,
            'scene_name': scene_name,
        }
        return ret
