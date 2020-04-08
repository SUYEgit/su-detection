# -*- coding: utf-8 -*-
"""
# Resnet Implementation
# Authors: suye
# Date: 2020/03/26
"""
import torch
import torch.nn as nn
from torchsummary import summary


class BuildingBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super(BuildingBlock, self).__init__()
        self.channels = channels
        self.out_channels = channels

        # output_shape = (image_shape-filter_shape+2*padding)/stride + 1
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.identity = self._prepare_identity(stride)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(channels)

    def forward(self, x):
        xf = self.conv1(x)
        xf = self.batchnorm(xf)
        xf = self.relu(xf)
        xf = self.conv2(xf)
        xf = self.batchnorm(xf)

        if self.identity:
            x = self.identity(x)
        return self.relu(x + xf)

    def _prepare_identity(self, stride):
        if stride == 1:
            return None
        elif stride == 2:
            conv_dot = nn.Conv2d(self.channels // 2, self.channels, kernel_size=1, stride=stride, bias=False)
            return nn.Sequential(conv_dot, nn.BatchNorm2d(self.channels))
        else:
            raise ValueError('Block got convolution stride {}'.format(stride))


class BottleNeck(nn.Module):
    def __init__(self, in_channels, base_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.base_channels = base_channels
        self.in_channels = in_channels
        self.out_channels = base_channels * 4

        # output_shape = (image_shape-filter_shape+2*padding)/stride + 1
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, bias=False)
        # stride在此处，原因之一为kernel 3 + padding 1可保证宽高不变（kernel 1就不一定了）
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(base_channels, self.out_channels, kernel_size=1, bias=False)

        self.identity = self._prepare_identity(stride)

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(base_channels)
        self.batchnorm1 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        xf = self.conv1(x)
        xf = self.batchnorm(xf)
        xf = self.relu(xf)
        xf = self.conv2(xf)
        xf = self.batchnorm(xf)
        xf = self.relu(xf)
        xf = self.conv3(xf)
        xf = self.batchnorm1(xf)

        if self.out_channels != self.in_channels:
            x = self.identity(x)

        return self.relu(xf + x)

    def _prepare_identity(self, stride):
        conv = nn.Conv2d(self.in_channels, self.base_channels * 4, kernel_size=1, stride=stride, bias=False)
        return nn.Sequential(conv, nn.BatchNorm2d(self.base_channels * 4))


settings = {
    18: (BuildingBlock, (2, 2, 2, 2)),
    34: (BuildingBlock, (3, 4, 6, 3)),
    50: (BottleNeck, (3, 4, 6, 3)),
    101: (BottleNeck, (3, 4, 23, 3)),
    152: (BottleNeck, (3, 8, 36, 3))
}


class ResNet(nn.Module):
    # TODO 用expansion优化make_res_layer部分
    # TODO 模型参数初始化（参照torch vision）
    def __init__(self, depth=34, num_classes=2, input_c=3):
        super(ResNet, self).__init__()

        self.base_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(input_c, self.base_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(self.base_channels),
                                   nn.ReLU())
        self.maxpool = nn.MaxPool2d(2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer_names = []
        for i in range(4):
            if depth > 34:
                res_layer = self.make_res_layer_by_bottleneck(block=settings[depth][0],
                                                              block_num=settings[depth][1][i],
                                                              stage=i + 1)
            else:
                res_layer = self.make_res_layer_by_buildblock(block=settings[depth][0],
                                                              block_num=settings[depth][1][i],
                                                              stage=i + 1)

            layer_name = 'res layer {}'.format(i + 1)
            self.add_module(layer_name, res_layer)  # 在循环中替代了逐一进行属性命名的过程
            self.layer_names.append(layer_name)

        self.linear = nn.Linear(res_layer[-1].out_channels, num_classes)

    def make_res_layer_by_bottleneck(self, block, block_num, stage):
        layers = []
        for i in range(block_num):
            channels = self.base_channels * (2 ** (stage - 1))
            if i != 0:
                layers.append(block(channels * 4, channels, stride=1))
            elif stage == 1:
                layers.append(block(channels, channels, stride=1))
            else:
                # 第2，3，4 stage的第一个下采样block
                layers.append(block(channels * 2, channels, stride=2))

        return nn.Sequential(*layers)

    def make_res_layer_by_buildblock(self, block, block_num, stage):
        layers = []
        for i in range(block_num):
            channels = self.base_channels * (2 ** (stage - 1))
            if stage > 1 and i == 0:
                # 第2，3，4 stage的第一个下采样block
                layers.append(block(channels // 2, channels, stride=2))
            else:
                layers.append(block(channels, channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        for layer_name in self.layer_names:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    resnet = ResNet(depth=152, num_classes=1000)

    resnet.cuda()
    input_size = 224
    summary(resnet, (3, input_size, input_size))
