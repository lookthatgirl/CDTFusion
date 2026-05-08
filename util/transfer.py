""" 
跨域迁移模块 (Cross-Domain Transfer Module)

本文件实现了两个模态（可见光/红外）之间的特征迁移机制。

包含:
- transfer: 基于通道注意力的跨域特征迁移模块
- SELayer: Squeeze-and-Excitation 通道注意力模块（已定义但未使用）

核心思想: 将两个模态的特征拼接后生成通道注意力权重，
用于加权调制另一个模态的特征，从而实现跨域信息迁移。
"""

import torch
import torch.nn as nn


class transfer(nn.Module):
    """
    跨域特征迁移模块 (Cross-Domain Transfer)

    将两个模态的特征进行交互融合，实现跨域信息迁移。
    例如 transfer_vi_to_ir 会将可见光特征迁移到红外域。

    工作流程:
    1. 拼接两个模态特征 (opt, sar) -> 1024通道
    2. 卷积降维到512通道，生成融合特征
    3. 通过1x1卷积生成通道注意力权重
    4. 用注意力权重加权调制sar特征
    5. 将调制后的特征与opt特征拼接，再次卷积融合

    输入: opt [B, 512, H, W], sar [B, 512, H, W]
    输出: [B, 512, H, W]
    """
    def __init__(self):
        super(transfer, self).__init__()

        # 第一组卷积: 拼接后的特征降维 1024 -> 1024 -> 512 -> 512
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)  # 特征拼接操作

        # 通道注意力: 生成通道维度的权重
        self.channel_weights_conv = nn.Conv2d(512, self.c, kernel_size=1)

        self.concat_x2 = lambda x: torch.cat(x, dim=1)  # 第二次拼接操作

        # 第二组卷积: 用于加权后的特征融合
        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, opt, sar):
        """
        前向传播

        参数:
            opt: 源模态特征 [B, 512, H, W]
            sar: 目标模态特征 [B, 512, H, W]

        返回:
            迁移后的融合特征 [B, 512, H, W]
        """
        # 步骤1: 拼接两个模态特征并卷积降维
        x1 = self.concat([opt, sar])          # [B, 1024, H, W]

        x1 = self.relu1(self.conv1(x1))       # [B, 1024, H, W]
        x1 = self.relu2(self.conv2(x1))       # [B, 512, H, W]
        x1 = self.relu3(self.conv3(x1))       # [B, 512, H, W]

        # 步骤2: 生成通道注意力权重，加权调制sar特征
        channel_weights = self.channel_weights_conv(x1)  # [B, 512, H, W]
        x2 = sar * channel_weights            # 通道注意力加权

        # 步骤3: 将调制后的特征与opt拼接，再次卷积融合
        x3 = self.concat_x2([opt, x2])        # [B, 1024, H, W]

        x3 = self.relu4(self.conv1(x3))       # [B, 1024, H, W]
        x3 = self.relu5(self.conv2(x3))       # [B, 512, H, W]
        x3 = self.relu6(self.conv3(x3))       # [B, 512, H, W]

        return x3


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块

    通过全局平均池化压缩空间维度，然后用全连接层学习通道间的依赖关系，
    最后对原始特征进行通道维度的重标定。

    参数:
        channel: 输入通道数
        reduction: 压缩比例，默认16
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
