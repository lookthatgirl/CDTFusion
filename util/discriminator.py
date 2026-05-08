""" 
域判别器模块 (Domain Discriminator Module)

本文件实现了用于对抗训练的域判别器。

域判别器的作用是区分特征来自可见光域还是红外域。
通过对抗训练，编码器被迫学习域不变的特征表示，
从而促进两个模态特征空间的对齐。
"""

import torch.nn as nn


class DomDiscriminator(nn.Module):
    """
    域判别器 (Domain Discriminator)

    用于判断输入特征来自可见光域还是红外域。
    在对抗训练中，判别器尝试区分两个域的特征，
    而编码器尝试欺骗判别器，从而学习域不变的特征。

    网络结构:
        4层卷积 (512->256->128->64->32，每层stride=2下采样)
        + 全连接层 (32*4*4 -> 128 -> 1)
        + Sigmoid 输出域标签 [0, 1]

    输入: [B, 512, 64, 64] 编码器输出的特征图
    输出: [B, 1] 域判别概率 (1=可见光域, 0=红外域)

    参数:
        input_channels: 输入特征通道数，默认512
    """
    def __init__(self, input_channels=512):
        super(DomDiscriminator, self).__init__()

        # 卷积层: 逐步下采样并降低通道数
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 全连接层: 将特征图展平后映射到单一域标签
        self.fc = nn.Sequential(
            nn.Flatten(),                        # [B, 32, 4, 4] -> [B, 512]
            nn.Linear(32 * 4 * 4, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),                   # -> [B, 1]
            nn.Sigmoid()                          # 输出 [0, 1] 概率
        )

    def forward(self, x):
        """ 输入特征图，输出域判别概率 """
        x = self.conv_layers(x)
        return self.fc(x)
