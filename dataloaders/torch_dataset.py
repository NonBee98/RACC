import json
import os
import random

import kornia
import numpy as np
import torch

from params import *
from utils import *

EPS = 1e-9


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root_dir,
                 extension='pt',
                 mode='train',
                 transform=None,
                 cross_validation=False,
                 fold_num=3,
                 fold_index=0,
                 random_seed=0,
                 shuffle=False,
                 test_ratio=0.2,
                 **kwargs):

        assert mode in ['train', 'val', 'test',
                        'all'], "Invalid mode: {}".format(mode)
        assert fold_index < fold_num, "Invalid fold index: {}".format(
            fold_index)
        if not isinstance(root_dir, (list, tuple)):
            root_dir = [root_dir]
        self.mode = mode
        self.data_list = []
        self.gt_list = []
        self.test_ratio = test_ratio
        file_list = []
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
        self.transform = transform

    def extract_features(self, ret):
        ret['target'] = ret['illum']
        return ret

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        gt_path = self.gt_list[index]
        with open(gt_path, 'r') as f:
            gt_dict = json.load(f)
        illum = torch.tensor(gt_dict['white_point'], dtype=torch.float32)
        als = torch.tensor(gt_dict['als'], dtype=torch.float32)
        als = als / (als.max() + 1e-8)
        img = torch.load(img_path, weights_only=True).float()
        img = img.permute(2, 0, 1)

        ret = {'illum': illum, 'input': img, 'extra_input': als}
        if self.transform is not None:
            ret = self.transform(ret)
        ret = self.extract_features(ret)
        return ret

    def worker_init_fn(self, worker_id):
        seed = torch.initial_seed()
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    def split_fold(self, data_list, fold_num, fold_index):
        test_num = int(len(data_list) * self.test_ratio)
        test_list = data_list[:test_num]
        data_list = data_list[test_num:]
        if self.mode == 'test':
            return test_list

        folds = []
        fold_size = len(data_list) // fold_num
        for i in range(fold_num):
            start_index = i * fold_size
            end_index = (i + 1) * fold_size
            if i == fold_num - 1:
                end_index = len(data_list)
            folds.append(data_list[start_index:end_index])
        if self.mode == 'val':
            return folds[fold_index]

        remain_index = [i for i in range(fold_num) if i != fold_index]
        ret = []
        for i in remain_index:
            ret += folds[i]
        return ret


class PCCdataset(TorchDataset):

    def extract_features(self, ret):
        """
        The four feature selected, i.e., bright, max, mean and dark pixels
        """
        img = ret["input"]
        feature_data = extract_statistical_features(
            img,
            channel_first=True,
            thresh_dark=PCCParams.thresh_dark,
            thresh_saturation=PCCParams.thresh_saturation,
        )

        ret["input"] = feature_data
        ret["target"] = ret["illum"]

        return ret


class RACCDataset(TorchDataset):

    def __init__(self,
                 root_dir,
                 extension='pt',
                 mode='train',
                 transform=None,
                 **kwargs):
        super().__init__(root_dir, extension, mode, transform, **kwargs)
        self.bin_num = CustomParams.bin_num
        self.boundary_value = CustomParams.boundary_value

        u_coord, v_coord = get_uv_coord(
            self.bin_num, range=self.boundary_value * 2
        )  # uv could be negative, range from -boundary_value to +boundary_value
        uv = torch.stack([u_coord, v_coord], dim=-1)
        self.coords_map = (uv + self.boundary_value) / (2 *
                                                        self.boundary_value)
        self.rgb_map = log_uv_to_rgb_torch(uv)

        self.rgb_map /= (torch.norm(
            self.rgb_map, dim=-1, keepdim=True, dtype=self.rgb_map.dtype) +
                         EPS)

    def extract_features(self, ret):
        img = ret['input']
        illum = ret['illum']

        hist = compute_uv_histogram_torch(
            img,
            self.bin_num,
            self.boundary_value,
            channel_first=True,
        )

        hist = torch.unsqueeze(hist, dim=0)
        ret['input'] = hist

        if CustomParams.edge_info:
            edge_img = kornia.filters.sobel(img.unsqueeze(0)).squeeze()
            if CustomParams.color_space == 'log_uv':
                edge_hist = compute_uv_histogram_torch(edge_img,
                                                       self.bin_num,
                                                       self.boundary_value,
                                                       channel_first=True)
            else:
                edge_hist = compute_chroma_histogram_torch(edge_img,
                                                           self.bin_num,
                                                           channel_first=True)
            edge_hist = torch.unsqueeze(edge_hist, dim=0)
            ret['input'] = torch.cat([ret['input'], edge_hist], dim=0)

        if CustomParams.coords_map:
            ret['input'] = torch.cat(
                [ret['input'], self.coords_map.permute(2, 0, 1)], dim=0)

        ret['target'] = illum
        return ret


class C5Dataset(TorchDataset):

    def __init__(
        self, root_dir, extension="pt", mode="train", transform=None, **kwargs
    ):
        super().__init__(root_dir, extension, mode, transform, **kwargs)
        self.bin_num = 64
        self.boundary_value = 2

        u_coord, v_coord = self.get_uv_coord(self.bin_num)
        self.coords_map = torch.concat([u_coord, v_coord], dim=0)

    def get_uv_coord(self, hist_size=64, tensor=True, normalize=False):
        """Gets uv-coordinate extra channels to augment each histogram as
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
            np.arange((hist_size - 1) / 2, (-(hist_size - 1) / 2) - 1, -1),
        )
        if normalize:
            u_coord = (u_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
            v_coord = (v_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
        if tensor:
            u_coord = torch.from_numpy(u_coord).to(dtype=torch.float32)
            u_coord = torch.unsqueeze(u_coord, dim=0)
            v_coord = torch.from_numpy(v_coord).to(dtype=torch.float32)
            v_coord = torch.unsqueeze(v_coord, dim=0)
        return u_coord, v_coord

    def extract_features(self, ret):
        img = ret["input"]
        illum = ret["illum"]
        hist = compute_uv_histogram_torch(
            img,
            self.bin_num,
            self.boundary_value,
            channel_first=True,
        )

        hist = torch.unsqueeze(hist, dim=0)
        ret["input"] = hist

        edge_img = kornia.filters.sobel(img.unsqueeze(0)).squeeze()

        edge_hist = compute_uv_histogram_torch(
            edge_img, self.bin_num, self.boundary_value, channel_first=True
        )

        edge_hist = torch.unsqueeze(edge_hist, dim=0)
        ret["input"] = torch.cat([ret["input"], edge_hist], dim=0)
        ret["input"] = torch.cat([ret["input"], self.coords_map], dim=0)

        ret["target"] = illum
        return ret
