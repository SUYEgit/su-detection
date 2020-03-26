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
    def __init__(self, channels):
        super(BuildingBlock, self).__init__()
        # output_shape = (image_shape-filter_shape+2*padding)/stride + 1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(channels)

    def forward(self, x):
        xf = self.conv1(x)
        xf = self.batchnorm(xf)
        xf = self.relu(xf)
        xf = self.conv2(xf)
        xf = self.batchnorm(xf)

        return x + xf


class BottleNeck(nn.Module):
    def __init__(self, channels):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(channels)

    def forward(self, x):
        xf = self.relu(self.batchnorm(self.conv1(x)))
        xf = self.relu(self.batchnorm(self.conv2(xf)))
        xf = self.conv3(xf)
        return xf + x


class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownsampleBlock, self).__init__()
        # 这里identity不做padding 主路线做2次1 padding 保证宽高一致
        self.conv1 = nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_dot = nn.Conv2d(channels // 2, channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(channels)

    def forward(self, x):
        print(x.shape)
        xf = self.conv1(x)
        xf = self.batchnorm(xf)
        xf = self.relu(xf)
        xf = self.conv2(xf)
        xf = self.batchnorm(xf)

        x = self.conv_dot(x)
        return x + xf


def make_res_layer(block, block_num, stage, base_channels=64):
    layers = []
    for i in range(block_num):
        channels = base_channels * (2 ** (stage - 1))
        if stage > 1 and i == 0:
            layers.append(DownsampleBlock(channels))
        else:
            layers.append(block(channels))
    return nn.Sequential(*layers)


settings = {
    18: (BuildingBlock, (2, 2, 2, 2)),
    34: (BuildingBlock, (3, 4, 6, 3)),
    50: (BottleNeck, (3, 4, 6, 3)),
    101: (BottleNeck, (3, 4, 23, 3)),
    152: (BottleNeck, (3, 8, 36, 3))
}


class ResNet(nn.Module):
    # TODO: 测试bottleneck
    # TODO: 实现DS bottleneck
    # TODO: 看一下downsample部分torchvision的官方实现
    def __init__(self, depth=34, num_classes=2):
        super(ResNet, self).__init__()
        self.layer_names = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

        for i in range(4):
            res_layer = make_res_layer(settings[depth][0], settings[depth][1][i], stage=i + 1)
            layer_name = 'res layer {}'.format(i + 1)
            self.add_module(layer_name, res_layer)  # 在循环中替代了逐一进行属性命名的过程
            self.layer_names.append(layer_name)

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
    resnet = ResNet(34)
    resnet.cuda()
    summary(resnet, (3, 224, 224))
