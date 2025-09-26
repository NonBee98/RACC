import torch
import torch.nn as nn

from params import *
from utils import *
from torch.nn import init
import random

from .common_blocks import *

__all__ = ["AUCC", "RACC", "racc_loss", "ALSOnly", "ImgOnly"]


def racc_loss(inputs: dict, targets: torch.Tensor) -> torch.Tensor:
    main_pred = inputs["main_output"]
    als_pred = inputs["als_auxiliary_output"]
    color_pred = inputs["color_auxiliary_output"]
    compressed_color_feature = inputs["compressed_color_feature"]
    compressed_als_data = inputs["compressed_als_data"]

    main_ae = angular_error_torch(main_pred, targets)
    als_ae = angular_error_torch(als_pred, targets)
    color_ae = angular_error_torch(color_pred, targets)

    regularization_loss = compressed_color_feature.mean() + compressed_als_data.mean()

    loss = (main_ae + als_ae + color_ae) / 3.0 + regularization_loss * 1e-2
    return loss


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=16):
        super(ConvEncoder, self).__init__()
        self.layer1 = Conv(in_channels, in_channels, 3, 1, norm="bn")
        self.layer2 = nn.Sequential(nn.AvgPool2d(2), Conv(in_channels, 4, 3, 1))
        self.layer3 = nn.Sequential(nn.AvgPool2d(2), Conv(4, 8, 3, 1, norm="bn"))
        self.layer4 = nn.Sequential(nn.AvgPool2d(2), Conv(8, 16, 3, 1, norm="bn"))
        self.layer5 = nn.Sequential(nn.AvgPool2d(2), Conv(16, 16, 3, 1, norm="bn"))
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, out_channels)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RepConvEncoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=16, dropout=0.0):
        super(RepConvEncoder, self).__init__()
        self.layer1 = RepConv(in_channels, in_channels, 3, 1, norm="bn")
        self.layer2 = nn.Sequential(nn.AvgPool2d(2), RepConv(in_channels, 4, 3, 1))
        self.layer3 = nn.Sequential(nn.AvgPool2d(2), RepConv(4, 8, 3, 1, norm="bn"))
        self.layer4 = nn.Sequential(
            nn.AvgPool2d(2), RepConv(8, 16, 3, 1, norm="bn", dropout=0.1)
        )
        self.layer5 = nn.Sequential(
            nn.AvgPool2d(2), RepConv(16, 16, 3, 1, norm="bn", dropout=0.1)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, out_channels)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()
        self.layer1 = Conv(in_channels, in_channels, 3, 1, norm="bn")
        self.layer2 = nn.Sequential(nn.AvgPool2d(2), Conv(in_channels, 4, 3, 1))
        self.layer3 = nn.Sequential(nn.AvgPool2d(2), Conv(4, 8, 3, 1, norm="bn"))
        self.layer4 = nn.Sequential(nn.AvgPool2d(2), Conv(8, 16, 3, 1, norm="bn"))
        self.layer5 = nn.Sequential(nn.AvgPool2d(2), Conv(16, 16, 3, 1, norm="bn"))

        self.upconv5 = nn.ConvTranspose2d(16, 16, 2, 2)
        self.uplayer4 = Conv(32, 16, 3, 1, norm="bn")

        self.upconv4 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.uplayer3 = Conv(16, 8, 3, 1, norm="bn")

        self.upconv3 = nn.ConvTranspose2d(8, 4, 2, 2)
        self.uplayer2 = Conv(8, 4, 3, 1, norm="bn")

        self.upconv2 = nn.ConvTranspose2d(4, 2, 2, 2)
        self.uplayer1 = Conv(in_channels + 2, in_channels, 3, 1, norm="bn")

        self.final_out = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x = self.upconv5(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.uplayer4(x)

        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.uplayer3(x)

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.uplayer2(x)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.uplayer1(x)

        x = self.final_out(x)
        return x


class UCC(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(UCC, self).__init__()
        u_coord, v_coord = get_uv_coord(
            CustomParams.bin_num, range=CustomParams.boundary_value * 2
        )
        uv_coords = torch.stack([u_coord, v_coord], dim=-1)
        self.rgb_map = log_uv_to_rgb_torch(uv_coords)  # h * w * 3
        self.rgb_map = self.rgb_map.reshape(-1, 3)
        if CustomParams.edge_info:
            in_channels += 1
        if CustomParams.coords_map:
            in_channels += 2
        self.model = UNet(in_channels, out_channels)
        self.init_params()

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.model(x)
        x = x.reshape(x.shape[0], -1)
        probability_map = F.softmax(x, dim=-1)
        probability_map = probability_map.reshape(-1, h, w)
        return probability_map

    def inference(self, x):
        self.rgb_map = self.rgb_map.to(x.device)
        probability_map = self.forward(x)
        b = probability_map.shape[0]
        probability_map = probability_map.reshape(b, -1)
        max_indexes = torch.argmax(probability_map, dim=-1)
        ret = self.rgb_map[max_indexes]
        return ret  # b * 3

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class AUCC(nn.Module):

    def __init__(self, in_channels=1, feature_dim=16, als_channels=9, out_features=2):
        super(AUCC, self).__init__()
        if CustomParams.edge_info:
            in_channels += 1
        if CustomParams.coords_map:
            in_channels += 2
        self.color_feature_encoder = ConvEncoder(in_channels, feature_dim)
        self.als_encoder = MLP(
            als_channels, out_features=feature_dim, dropout=0, layer_num=3
        )
        self.head = MLP(
            feature_dim, out_features=out_features, dropout=0.0, layer_num=5
        )

    def convert_rb_to_rgb(self, x):
        rg = x[..., 0:1]
        bg = x[..., 1:]
        ones = torch.ones_like(rg, dtype=rg.dtype, device=rg.device)
        output = torch.cat([rg, ones, bg], dim=-1)
        return output

    def forward(self, inputs):
        color_feature = inputs["input"]
        als_data = inputs["extra_input"]

        compressed_color_feature = self.color_feature_encoder(color_feature)
        compressed_als_data = self.als_encoder(als_data)
        # compressed_feature = compressed_color_feature + compressed_als_data
        compressed_feature = compressed_color_feature
        out = self.head(compressed_feature)
        output = self.convert_rb_to_rgb(out)
        return output

    def inference(self, inputs):
        return self.forward(inputs)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class RACC(nn.Module):

    def __init__(
        self,
        in_channels=1,
        feature_dim=16,
        als_channels=9,
        out_features=2,
        drop_path_rate=0.2,
    ):
        super(RACC, self).__init__()
        if CustomParams.edge_info:
            in_channels += 1
        if CustomParams.coords_map:
            in_channels += 2
        self.drop_path_rate = drop_path_rate

        self.color_feature_encoder = RepConvEncoder(in_channels, feature_dim)
        self.als_encoder = MLP(als_channels, out_features=feature_dim, layer_num=3)
        self.head = MLP(feature_dim, out_features=out_features, layer_num=5)

        self.color_auxiliary_head = MLP(
            feature_dim, out_features=out_features, layer_num=3
        )
        self.als_auxiliary_head = MLP(
            feature_dim, out_features=out_features, layer_num=3
        )

    def convert_rb_to_rgb(self, x):
        rg = x[..., 0:1]
        bg = x[..., 1:]
        ones = torch.ones_like(rg, dtype=rg.dtype, device=rg.device)
        output = torch.cat([rg, ones, bg], dim=-1)
        return output

    def forward(self, inputs):
        color_feature = inputs["input"]
        als_data = inputs["extra_input"]

        compressed_color_feature = self.color_feature_encoder(color_feature)
        compressed_als_data = self.als_encoder(als_data)
        compressed_als_data_weight = 1
        if random.random() < self.drop_path_rate:
            compressed_als_data_weight = 0
        compressed_feature = (
            compressed_color_feature + compressed_als_data * compressed_als_data_weight
        )
        main_output = self.convert_rb_to_rgb(self.head(compressed_feature))

        color_auxiliary_output = self.convert_rb_to_rgb(
            self.color_auxiliary_head(compressed_color_feature)
        )
        als_auxiliary_output = self.convert_rb_to_rgb(
            self.als_auxiliary_head(compressed_als_data)
        )

        ret = {
            "main_output": main_output,
            "color_auxiliary_output": color_auxiliary_output,
            "als_auxiliary_output": als_auxiliary_output,
            "compressed_color_feature": compressed_color_feature,
            "compressed_als_data": compressed_als_data,
        }
        return ret

    def inference(self, inputs):
        color_feature = inputs["input"]
        als_data = inputs["extra_input"]

        compressed_color_feature = self.color_feature_encoder(color_feature)
        compressed_als_data = self.als_encoder(als_data)
        compressed_feature = compressed_color_feature + compressed_als_data
        output = self.convert_rb_to_rgb(self.head(compressed_feature))
        return output

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def fuse_params(self):
        for module in self.modules():
            if isinstance(module, RepConv):
                module.fuse_params()
        return self

    def deploy(self):
        self.fuse_params()
        try:
            del self.als_auxiliary_head
            del self.color_auxiliary_head
        except:
            pass
        self.forward = self.inference
        return self


class ALSOnly(nn.Module):

    def __init__(self, feature_dim=16, als_channels=9, out_features=2):
        super(ALSOnly, self).__init__()
        self.als_encoder = MLP(
            als_channels, out_features=feature_dim, dropout=0, layer_num=3
        )
        self.head = MLP(
            feature_dim, out_features=out_features, dropout=0.0, layer_num=5
        )

    def convert_rb_to_rgb(self, x):
        rg = x[..., 0:1]
        bg = x[..., 1:]
        ones = torch.ones_like(rg, dtype=rg.dtype, device=rg.device)
        output = torch.cat([rg, ones, bg], dim=-1)
        return output

    def forward(self, inputs):
        als_data = inputs["extra_input"]

        compressed_als_data = self.als_encoder(als_data)
        compressed_feature = compressed_als_data
        out = self.head(compressed_feature)
        output = self.convert_rb_to_rgb(out)
        return output

    def inference(self, inputs):
        return self.forward(inputs)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ImgOnly(nn.Module):

    def __init__(self, in_channels=1, feature_dim=16, out_features=2):
        super(ImgOnly, self).__init__()
        if CustomParams.edge_info:
            in_channels += 1
        if CustomParams.coords_map:
            in_channels += 2
        self.color_feature_encoder = ConvEncoder(in_channels, feature_dim)
        self.head = MLP(
            feature_dim, out_features=out_features, dropout=0.0, layer_num=5
        )

    def convert_rb_to_rgb(self, x):
        rg = x[..., 0:1]
        bg = x[..., 1:]
        ones = torch.ones_like(rg, dtype=rg.dtype, device=rg.device)
        output = torch.cat([rg, ones, bg], dim=-1)
        return output

    def forward(self, inputs):
        color_feature = inputs["input"]
        compressed_color_feature = self.color_feature_encoder(color_feature)
        compressed_feature = compressed_color_feature
        out = self.head(compressed_feature)
        output = self.convert_rb_to_rgb(out)
        return output

    def inference(self, inputs):
        return self.forward(inputs)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
