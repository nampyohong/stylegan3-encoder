import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

import pdb


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))

    # num_layers -- 50
    return blocks


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        # pdb.set_trace()
        # channels = 64
        # reduction = 16

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )
        # in_channel = 64
        # depth = 64
        # stride = 2

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        # in_c = 512
        # out_c = 512
        # spatial = 16

        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        # num_pools -- 4
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)
        # self = GradualStyleBlock(
        #   (convs): Sequential(
        #     (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (1): LeakyReLU(negative_slope=0.01)
        #     (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (3): LeakyReLU(negative_slope=0.01)
        #     (4): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (5): LeakyReLU(negative_slope=0.01)
        #     (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #     (7): LeakyReLU(negative_slope=0.01)
        #   )
        #   (linear): Linear(in_features=512, out_features=512, bias=True)
        # )

    def forward(self, x):
        x = self.convs(x)  # [b,512,H,W]->[b,512,1,1]
        # (H,W) in [(16,16),(32,32),(64,64)]
        x = x.view(-1, self.out_c)  # [b,512,1,1]-> [b,512]
        x = self.linear(x)
        x = nn.LeakyReLU()(x)
        return x


class GradualStyleEncoder(nn.Module):
    def __init__(self):
        super(GradualStyleEncoder, self).__init__()

        # Create ArcFace Model
        blocks = get_blocks(50)  # num_layers=50
        unit_module = bottleneck_IR_SE  # 'ir_se' bottleneck

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )  # [b,3,256,256]->[b,3,64,64]

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = nn.Sequential(*modules)

        # Create Style Model
        self.styles = nn.ModuleList()  # feat->latent

        # TODO:
        # need some other method for handling w[0]
        # train w[0] separately ?
        # coarse_ind, middle_ind tuning
        self.style_count = 16
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)  # [b,3,256,256]->[b,64,256,256]

        latents = []
        modulelist = list(self.body._modules.values())

        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x  # [b,128,64,64]
            elif i == 20:
                c2 = x  # [b,256,32,32]
            elif i == 23:
                c3 = x  # [b,512,16,16]

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))
        # (Pdb) len(latents) -- 3, latents[0..2].size() -- [4, 512]

        p2 = self._upsample_add(c3, self.latlayer1(c2))  # [b,512,32,32]
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))
        # (Pdb) len(latents) -- 7, latents[3..6].size() -- [4, 512]

        p1 = self._upsample_add(p2, self.latlayer2(c1))  # [b,512,64,64]
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
        # (Pdb) len(latents) -- 16, latents[7..15].size() -- [4, 512]
        out = torch.stack(latents, dim=1)  # [4, 16, 512]

        return out
