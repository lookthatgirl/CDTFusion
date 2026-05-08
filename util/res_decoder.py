"""
解码器头模块 (Decoder Head Module)

本文件实现了多种解码器头，用于将编码器提取的特征图解码为不同类型的输出。

解码器头分为三类:
1. 重建解码器 (用于Step1训练):
   - head_1: 输出3通道RGB图像（用于重建可见光图像）
   - head_2: 输出1通道灰度图像（用于重建红外图像）
   - head_3: 输出3通道（与head_1相同，未使用）

2. 融合解码器 (用于Step2训练):
   - head_fus: 输出1通道融合灰度图像

3. 分割解码器 (用于Step2训练，不同数据集使用不同的头):
   - head_seg: 6类分割（基础版）
   - head_seg_dice: 6类分割（带额外卷积）
   - head_seg_sar_dice: 8类分割
   - head_seg_pos: 6类分割（Potsdam数据集，带跳跃连接）
   - head_seg_whu: 8类分割（WHU数据集，带跳跃连接）
   - head_seg_fmb: 15类分割（FMB数据集，带跳跃连接）

所有解码器都采用转置卷积(ConvTranspose2d)进行上采样，结合残差块进行特征细化。
输入: [B, 512, H/8, W/8] 编码器特征
输出: [B, C, H, W] 其中C取决于具体的解码器头类型
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch


class ResidualBlock(nn.Module):
    """
    解码器用残差块 (Decoder Residual Block)

    与编码器中的残差块类似，但使用 LeakyReLU(0.1) 代替 ReLU，
    并使用 Xavier 初始化卷积权重。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.relu2 = nn.LeakyReLU(0.1)

        # Xavier初始化权重，有助于训练稳定性
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)

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
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out


class head_1(nn.Module):
    """
    可见光重建解码器 (Visible Image Reconstruction Decoder)

    将512通道特征图解码为3通道RGB图像。
    用于Step1训练中，将IR->VI迁移后的特征重建为可见光图像。

    结构: ConvT(512->256) -> ResBlock -> ConvT(256->128) -> ResBlock
           -> ConvT(128->64) -> ResBlock -> Conv(64->3) -> ResBlock

    输入: [B, 512, H/8, W/8]
    输出: [B, 3, H, W]
    """
    def __init__(self):
        super(head_1, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(3, 3)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)


    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)

        x = self.upconv3(x)
        x = self.residual3(x)

        x = self.upconv2(x)
        x = self.residual2(x)

        x = self.conv1(x)
        x = self.residual1(x)

        return x

class head_2(nn.Module):
    """
    红外重建解码器 (Infrared Image Reconstruction Decoder)

    将512通道特征图解码为1通道灰度图像。
    用于Step1训练中，将VI->IR迁移后的特征重建为红外图像。

    输入: [B, 512, H/8, W/8]
    输出: [B, 1, H, W]
    """
    def __init__(self):
        super(head_2, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)

        self.conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(1, 1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)


    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)

        x = self.upconv3(x)
        x = self.residual3(x)

        x = self.upconv2(x)
        x = self.residual2(x)

        x = self.conv1(x)
        x = self.residual1(x)

        return x

class head_3(nn.Module):
    """
    备用重建解码器 (Backup Reconstruction Decoder)

    与head_1结构完全相同，输出3通道。未在当前训练流程中使用。
    """
    def __init__(self):
        super(head_3, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(3, 3)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)



    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)

        x = self.upconv3(x)
        x = self.residual3(x)

        x = self.upconv2(x)
        x = self.residual2(x)

        x = self.conv1(x)
        x = self.residual1(x)

        return x

class head_seg(nn.Module):
    """
    基础分割解码器 (Basic Segmentation Decoder)

    输出6类分割结果，带softmax激活。未在当前训练流程中直接使用。

    输入: [B, 512, H/8, W/8]
    输出: [B, 6, H, W] 每个像素6类的概率
    """
    def __init__(self):
        super(head_seg, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)

        self.conv1 = nn.Conv2d(64, 6, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(6, 6)
        # self.conv = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)

    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)

        x = self.upconv3(x)
        x = self.residual3(x)

        x = self.upconv2(x)
        x = self.residual2(x)

        x = self.conv1(x)
        x = self.residual1(x)
        # x = self.conv(x)

        x = F.softmax(x, dim=1)

        return x

class head_fus(nn.Module):
    """
    融合解码器 (Fusion Decoder)

    将融合特征解码为1通道灰度融合图像。
    用于Step2训练，生成的灰度图将替换原图HSV的V通道以生成彩色融合图。

    输入: [B, 512, H/8, W/8]
    输出: [B, 1, H, W]
    """
    def __init__(self):
        super(head_fus, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)

        self.conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(1, 1)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)


    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)

        x = self.upconv3(x)
        x = self.residual3(x)

        x = self.upconv2(x)
        x = self.residual2(x)

        x = self.conv1(x)
        x = self.residual1(x)
        # x = self.conv(x)

        return x

class head_seg_dice(nn.Module):
    """
    Dice损失分割解码器 (Dice Loss Segmentation Decoder)

    6类分割，带额外的3x3卷积和softmax。未在当前训练流程中直接使用。

    输入: [B, 512, H/8, W/8]
    输出: [B, 6, H, W]
    """
    def __init__(self):
        super(head_seg_dice, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)

        self.conv1 = nn.Conv2d(64, 6, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(6, 6)
        self.conv = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)

        x = self.upconv3(x)
        x = self.residual3(x)

        x = self.upconv2(x)
        x = self.residual2(x)

        x = self.conv1(x)
        x = self.residual1(x)

        x = self.conv(x)
        x = F.softmax(x, dim=1)

        return x

class head_seg_sar_dice(nn.Module):
    """
    8类分割解码器 (8-Class Segmentation Decoder)

    8类分割，带额外卸3x3卷积和softmax。未在当前训练流程中直接使用。

    输入: [B, 512, H/8, W/8]
    输出: [B, 8, H, W]
    """
    def __init__(self):
        super(head_seg_sar_dice, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)

        self.conv1 = nn.Conv2d(64, 8, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(8, 8)
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.upconv4(x)
        x = self.residual4(x)

        x = self.upconv3(x)
        x = self.residual3(x)

        x = self.upconv2(x)
        x = self.residual2(x)

        x = self.conv1(x)
        x = self.residual1(x)

        x = self.conv(x)
        x = F.softmax(x, dim=1)

        return x


class head_seg_pos(nn.Module):
    """
    Potsdam数据集分割解码器 (POS Segmentation Decoder)

    6类分割，带跳跃连接(skip connection)。
    每个上采样阶段都将上一层特征通过双线性插值上采样后与当前层拼接，
    并通过3x3卷积降维，有助于保留多尺度的空间细节。

    输入: [B, 512, H/8, W/8]
    输出: [B, 6, H, W] (ImSurf, Building, LowVeg, Tree, Car, Clutter)
    """
    def __init__(self):
        super(head_seg_pos, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)
        self.bn1 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)
        self.bn2 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 6, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(6, 6)
        self.conv = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv.weight)

        self.conv_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(192, 64, kernel_size=3, padding=1)

    def forward(self, x):
        # 跳跃连接: 双线性插值上采样 + 拼接 + 卷积降维
        x1 = x
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')  # [B, 512, H/4, W/4]

        x = self.upconv4(x)                  # [B, 256, H/4, W/4]
        x = torch.cat((x,x1), dim=1)         # [B, 768, H/4, W/4] 跳跃连接
        x = self.conv_1(x)                   # [B, 256, H/4, W/4] 降维

        x = self.residual4(x)
        x = self.bn1(x)
        x2 = F.interpolate(x, scale_factor=2, mode='bilinear')  # [B, 256, H/2, W/2]

        x = self.upconv3(x)                  # [B, 128, H/2, W/2]
        x = torch.cat((x,x2), dim=1)         # [B, 384, H/2, W/2] 跳跃连接
        x = self.conv_2(x)                   # [B, 128, H/2, W/2]

        x = self.residual3(x)
        x = self.bn2(x)
        x3 = F.interpolate(x, scale_factor=2, mode='bilinear')  # [B, 128, H, W]

        x = self.upconv2(x)                  # [B, 64, H, W]
        x = torch.cat((x,x3), dim=1)         # [B, 192, H, W] 跳跃连接
        x = self.conv_3(x)                   # [B, 64, H, W]

        x = self.residual2(x)
        x = self.bn3(x)

        x = self.conv1(x)                    # [B, 6, H, W]
        x = self.residual1(x)

        x = self.conv(x)
        x = F.softmax(x, dim=1)              # 输出每个像素6类的概率

        return x


class head_seg_whu(nn.Module):
    """
    WHU数据集分割解码器 (WHU Segmentation Decoder)

    8类分割，带跳跃连接。结构与head_seg_pos相同，仅输出类别数不同。

    输入: [B, 512, H/8, W/8]
    输出: [B, 8, H, W] (Background, Farmland, City, Village, Water, Forest, Road, Other)
    """
    def __init__(self):
        super(head_seg_whu, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)
        self.bn1 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)
        self.bn2 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 8, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(8, 8)
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv.weight)

        self.conv_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(192, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = x
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')

        x = self.upconv4(x)
        x = torch.cat((x,x1), dim=1)
        x = self.conv_1(x)

        x = self.residual4(x)
        x = self.bn1(x)
        x2 = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.upconv3(x)
        x = torch.cat((x,x2), dim=1)
        x = self.conv_2(x)

        x = self.residual3(x)
        x = self.bn2(x)
        x3 = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.upconv2(x)
        x = torch.cat((x,x3), dim=1)
        x = self.conv_3(x)

        x = self.residual2(x)
        x = self.bn3(x)

        x = self.conv1(x)
        x = self.residual1(x)

        x = self.conv(x)
        x = F.softmax(x, dim=1)

        return x

class head_seg_fmb(nn.Module):
    """
    FMB数据集分割解码器 (FMB Segmentation Decoder)

    15类分割，带跳跃连接。结构与head_seg_pos相同，仅输出类别数不同。

    输入: [B, 512, H/8, W/8]
    输出: [B, 15, H, W]
    """
    def __init__(self):
        super(head_seg_fmb, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(256, 256)
        self.bn1 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(128, 128)
        self.bn2 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(64, 64)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 15, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(15, 15)
        self.conv = nn.Conv2d(15, 15, kernel_size=3, stride=1, padding=1)

        init.xavier_uniform_(self.upconv4.weight)
        init.xavier_uniform_(self.upconv3.weight)
        init.xavier_uniform_(self.upconv2.weight)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv.weight)

        self.conv_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(192, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = x
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')

        x = self.upconv4(x)
        x = torch.cat((x,x1), dim=1)
        x = self.conv_1(x)

        x = self.residual4(x)
        x = self.bn1(x)
        x2 = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.upconv3(x)
        x = torch.cat((x,x2), dim=1)
        x = self.conv_2(x)

        x = self.residual3(x)
        x = self.bn2(x)
        x3 = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.upconv2(x)
        x = torch.cat((x,x3), dim=1)
        x = self.conv_3(x)

        x = self.residual2(x)
        x = self.bn3(x)

        x = self.conv1(x)
        x = self.residual1(x)

        x = self.conv(x)
        x = F.softmax(x, dim=1)

        return x
