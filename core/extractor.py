# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,logging,os,sys,urllib,warnings
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import numpy as np
from core.submodule import LayerNorm2d, BasicConv, Conv2x_IN
from Utils import get_resize_keep_aspect_ratio, freeze_model
import timm


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn=='layer':
            self.norm1 = LayerNorm2d(planes)
            self.norm2 = LayerNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = LayerNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn=='layer':
            self.norm1 = LayerNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []

        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)



class ContextNetDino(MultiBasicEncoder):
    def __init__(self, args, output_dim=[128], norm_fn='batch', downsample=3):
        nn.Module.__init__(self)
        self.args = args
        self.patch_size = 14
        self.image_size = 518
        self.vit_feat_dim = 384
        code_dir = os.path.dirname(os.path.realpath(__file__))

        self.out_dims = output_dim

        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn=='layer':
            self.norm1 = LayerNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        self.down = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0),
          nn.BatchNorm2d(128),
        )
        vit_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2
        self.conv2 = BasicConv(128+vit_dim, 128, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(256)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

    def forward(self, x_in, vit_feat, dual_inp=False, num_layers=3):
        B,C,H,W = x_in.shape
        x = self.conv1(x_in)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        divider = np.lcm(self.patch_size, 16)
        H_resize, W_resize = get_resize_keep_aspect_ratio(H,W, divider=divider, max_H=1344, max_W=1344)
        x = torch.cat([x, vit_feat], dim=1)
        x = self.conv2(x)
        outputs04 = [f(x) for f in self.outputs04]

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16)


class DepthAnythingFeature(nn.Module):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    def __init__(self, encoder='vits'):
        super().__init__()
        from depth_anything.dpt import DepthAnything
        self.encoder = encoder
        depth_anything = DepthAnything(self.model_configs[encoder])
        self.depth_anything = depth_anything

        self.intermediate_layer_idx = {   #!NOTE For V2
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }


    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        h, w = x.shape[-2:]
        features = self.depth_anything.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)


        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        out, path_1, path_2, path_3, path_4, disp = self.depth_anything.depth_head.forward(features, patch_h, patch_w, return_intermediate=True)

        return {'out':out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4, 'features':features, 'disp':disp}  # path_1 is 1/2; path_2 is 1/4


class Feature(nn.Module):
    def __init__(self, args):
        super(Feature, self).__init__()
        self.args = args
        model = timm.create_model('edgenext_small', pretrained=True, features_only=False)
        self.stem = model.stem
        self.stages = model.stages
        chans = [48, 96, 160, 304]
        self.chans = chans
        self.dino = DepthAnythingFeature(encoder=self.args.vit_size)
        self.dino = freeze_model(self.dino)
        vit_feat_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2

        self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)
        self.conv4 = nn.Sequential(
          BasicConv(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
        )

        self.patch_size = 14
        self.d_out = [chans[0]*2+vit_feat_dim, chans[1]*2, chans[2]*2, chans[3]]

    def forward(self, x):
        B,C,H,W = x.shape
        divider = np.lcm(self.patch_size, 16)
        H_resize, W_resize = get_resize_keep_aspect_ratio(H,W, divider=divider, max_H=1344, max_W=1344)
        x_in_ = F.interpolate(x, size=(H_resize, W_resize), mode='bicubic', align_corners=False)
        self.dino = self.dino.eval()
        with torch.no_grad():
          output = self.dino(x_in_)
        vit_feat = output['out']
        vit_feat = F.interpolate(vit_feat, size=(H//4,W//4), mode='bilinear', align_corners=True)
        x = self.stem(x)
        x4 = self.stages[0](x)
        x8 = self.stages[1](x4)
        x16 = self.stages[2](x8)
        x32 = self.stages[3](x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = torch.cat([x4, vit_feat], dim=1)
        x4 = self.conv4(x4)
        return [x4, x8, x16, x32], vit_feat


