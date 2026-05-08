"""
编码器模块 (Encoder Module)

本文件定义了红外-可见光图像融合框架中的特征编码器。
包含两个编码器：
- EncoderVi: 可见光图像编码器（输入3通道RGB图像）
- EncoderIr: 红外图像编码器（输入1通道灰度图像）

两个编码器结构相同，均使用残差块(ResidualBlock)提取多尺度特征，
经过3次最大池化下采样后输出512通道的特征图。
对于512x512的输入图像，输出特征图尺寸为64x64x512。
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)

    基本结构: Conv -> ReLU -> Conv -> 残差连接 -> ReLU
    当输入和输出通道数不同或步长不为1时，使用1x1卷积+BN进行下采样以匹配维度。

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 卷积步长，默认为1
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 第一层卷积: 可能改变通道数和空间尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.relu1 = nn.ReLU(inplace=True)
        # 第二层卷积: 保持通道数和空间尺寸不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # 当通道数变化或步长不为1时，需要对恒等映射进行下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        # 残差连接: 将输入直接加到输出上
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out

class EncoderVi(nn.Module):
    """
    可见光图像编码器 (Visible Image Encoder)

    输入: 3通道RGB可见光图像 [B, 3, H, W]
    输出: 512通道特征图 [B, 512, H/8, W/8]

    网络结构:
        Conv(3->64) -> ResBlock(64->64) -> MaxPool(2倍下采样)
        -> ResBlock(64->128) -> MaxPool(2倍下采样)
        -> ResBlock(128->256) -> MaxPool(2倍下采样)
        -> ResBlock(256->512)
    """
    def __init__(self):
        super(EncoderVi, self).__init__()

        # 初始卷积: 将3通道RGB输入映射到64通道特征
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # 第一阶段: 64通道残差提取 + 2倍下采样
        self.res_block1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二阶段: 通道扩展到128 + 2倍下采样
        self.res_block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三阶段: 通道扩展到256 + 2倍下采样
        self.res_block3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四阶段: 通道扩展到512（不再下采样）
        self.res_block4 = ResidualBlock(256, 512)

    def forward(self, x):
        x = self.relu1(self.conv1(x))   # [B, 64, H, W]

        x = self.res_block1(x)          # [B, 64, H, W]
        x = self.pool1(x)               # [B, 64, H/2, W/2]

        x = self.res_block2(x)          # [B, 128, H/2, W/2]
        x = self.pool2(x)               # [B, 128, H/4, W/4]

        x = self.res_block3(x)          # [B, 256, H/4, W/4]
        x = self.pool3(x)               # [B, 256, H/8, W/8]

        x = self.res_block4(x)          # [B, 512, H/8, W/8]

        return x

class EncoderIr(nn.Module):
    """
    红外图像编码器 (Infrared Image Encoder)

    输入: 1通道灰度红外图像 [B, 1, H, W]
    输出: 512通道特征图 [B, 512, H/8, W/8]

    网络结构与 EncoderVi 完全相同，仅输入通道数不同（1 vs 3）。
    """
    def __init__(self):
        super(EncoderIr, self).__init__()

        # 初始卷积: 将1通道红外输入映射到64通道特征
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # 后续结构与 EncoderVi 相同
        self.res_block1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_block4 = ResidualBlock(256, 512)

    def forward(self, x):
        x = self.relu1(self.conv1(x))   # [B, 64, H, W]

        x = self.res_block1(x)          # [B, 64, H, W]
        x = self.pool1(x)               # [B, 64, H/2, W/2]

        x = self.res_block2(x)          # [B, 128, H/2, W/2]
        x = self.pool2(x)               # [B, 128, H/4, W/4]

        x = self.res_block3(x)          # [B, 256, H/4, W/4]
        x = self.pool3(x)               # [B, 256, H/8, W/8]

        x = self.res_block4(x)          # [B, 512, H/8, W/8]

        return x
