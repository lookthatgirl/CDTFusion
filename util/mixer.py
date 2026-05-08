"""
特征混合器模块 (Feature Mixer Module)

本文件实现了多种特征混合器，用于融合两个模态的特征。

包含:
- mixer: 简单的拼接+卷积混合器
- FeedForward / MixerBlock / MLPMixer: 基于MLP-Mixer论文的实现（未使用）
- Mixer: 主要使用的混合器，结合通道混合和Token混合
- Mixer_resize: 支持可变输入尺寸的混合器变体
- Mixer_resize1: 使用GroupNorm替代LayerNorm的混合器变体

核心思想: 通过通道混合和Token混合两个分支，分别建模通道间和空间位置间的依赖关系。
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d
from einops.layers.torch import Rearrange
import numpy as np


class mixer(nn.Module):
    """
    简单特征混合器 (Simple Feature Mixer)

    将两个模态的特征拼接后通过卷积降维融合。
    结构: concat(1024ch) -> Conv(1024) -> Conv(512) -> Conv(512)

    输入: opt [B, 512, H, W], sar [B, 512, H, W]
    输出: [B, 512, H, W]
    """
    def __init__(self):
        super(mixer, self).__init__()

        # 拼接后的卷积降维: 1024 -> 1024 -> 512 -> 512
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)  # 特征拼接操作

    def forward(self, opt, sar):
        x1 = self.concat([opt, sar])       # [B, 1024, H, W]
        x1 = self.relu1(self.conv1(x1))    # [B, 1024, H, W]
        x1 = self.relu2(self.conv2(x1))    # [B, 512, H, W]
        x1 = self.relu3(self.conv3(x1))    # [B, 512, H, W]

        return x1


class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)

    两层全连接 + GELU激活，用于MLP-Mixer中的Token混合和通道混合。
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self,x):
        x = self.net(x)
        return x


class MixerBlock(nn.Module):
    """
    MLP-Mixer基本块

    包含两个子模块:
    - Token Mixer: 在空间维度上混合信息（跨patch交互）
    - Channel Mixer: 在通道维度上混合信息（跨特征交互）
    两者都使用残差连接。
    """
    def __init__(self, dim, num_patch, token_dim, channel_dim):
        super().__init__()
        # Token Mixer: 在patch维度上混合
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),       # 转置以在patch维度上操作
            FeedForward(num_patch, token_dim),
            Rearrange('b d n -> b n d')         # 转置回来

        )
        # Channel Mixer: 在通道维度上混合
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)    # Token混合 + 残差
        x = x + self.channel_mixer(x)  # 通道混合 + 残差
        return x


class MLPMixer(nn.Module):
    """
    MLP-Mixer模型 (基于MLP-Mixer论文的实现)

    将拼接后的特征分割为patch序列，然后通过多层MixerBlock进行混合。
    （已定义但未在当前训练流程中使用）

    参数:
        in_channels: 输入通道数
        dim: 嵌入维度
        patch_size: patch大小
        image_size: 输入图像尺寸
        depth: MixerBlock层数
        token_dim: Token混合的隐藏维度
        channel_dim: 通道混合的隐藏维度
    """
    def __init__(self, in_channels, dim, patch_size, image_size, depth, token_dim, channel_dim,
                 dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2

        # Patch嵌入: 将图像分割为patch并投影到嵌入空间
        self.to_embedding = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')  # 展平为patch序列
            )

        # 多层MixerBlock
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patches, token_dim, channel_dim))

        self.layer_normal = nn.LayerNorm(dim)

        # 拼接后的卷积降维
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)

    def forward(self, opt, sar):
        # 拼接 + 卷积降维
        x = self.concat([opt, sar])         # [B, 1024, H, W]
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))       # [B, 512, H, W]

        # Patch嵌入 + MixerBlock序列
        x = self.to_embedding(x)            # [B, num_patches, dim]
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_normal(x)

        # 恢复为2D特征图
        b, h_w, c = x.shape
        h = w = int(np.sqrt(h_w))
        x = x.reshape(b, c, h, w)           # [B, dim, h, w]

        return x




# ==================== 以下为实际使用的Mixer变体 ====================

import torch
import torch.nn as nn
from torch.nn import Conv2d
from einops.layers.torch import Rearrange
import numpy as np


class Mixer(nn.Module):
    """
    主特征混合器 (Main Feature Mixer) - 实际训练中使用的版本

    结合通道混合和Token混合两个分支，分别建模通道间和空间位置间的依赖关系。

    工作流程:
    1. 拼接两个模态特征 -> 卷积降维得到 x1
    2. 通道混合分支 (mix1): 对x1做1x1卷积 + LayerNorm + ReLU + 1x1卷积
    3. Token混合分支 (mix2): 对opt+sar的逐元素和做同样的变换
    4. GELU激活: x1 + x2 + channel_mixed + token_mixed
    5. 残差连接: 加上opt和sar

    参数:
        in_channels: 输入通道数 (512)
        out_channels: 输出通道数 (512)
        hidden_dim_channel: 通道混合的隐藏维度 (1024)
        hidden_dim_token: Token混合的隐藏维度 (512)

    注意: 内部LayerNorm固定了空间尺寸为64x64，因此仅适用于512x512的输入图像。
    """
    def __init__(self, in_channels, out_channels, hidden_dim_channel, hidden_dim_token):
        super(Mixer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim_channel = hidden_dim_channel
        self.hidden_dim_token = hidden_dim_token

        # 拼接后的卷积降维: 1024 -> 512
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)

        # 通道混合分支: 1x1卷积学习通道间的依赖关系
        self.mix1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_channel, kernel_size=1),
            nn.LayerNorm([hidden_dim_channel, 64, 64]),  # 固定空间尺寸64x64
            nn.ReLU(),
            nn.Conv2d(hidden_dim_channel, out_channels, kernel_size=1)
        )

        # Token混合分支: 1x1卷积学习空间位置间的依赖关系
        self.mix2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_token, kernel_size=1),
            nn.LayerNorm([hidden_dim_token, 64, 64]),    # 固定空间尺寸64x64
            nn.ReLU(),
            nn.Conv2d(hidden_dim_token, out_channels, kernel_size=1)
        )
        self.gelu = nn.GELU()

    def forward(self, opt, sar):
        # 拼接 + 卷积降维
        x1 = self.concat([opt, sar])               # [B, 1024, H, W]
        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))             # [B, 512, H, W]

        # 通道混合: 对拼接融合后的特征做通道混合
        channel_mixed = self.mix1(x1)               # [B, 512, H, W]

        # Token混合: 对逐元素相加的特征做Token混合
        x2 = opt + sar                              # [B, 512, H, W]
        token_mixed = self.mix2(x2)                 # [B, 512, H, W]

        # 融合所有分支 + GELU激活 + 残差连接
        mixed_features = self.gelu(x1 + x2 + channel_mixed + token_mixed)
        output = mixed_features + opt + sar
        return output


class Mixer_resize(nn.Module):
    """
    可变尺寸特征混合器 (Resizable Feature Mixer)

    与Mixer结构相同，但LayerNorm的空间尺寸通过参数x, y指定，
    支持不同尺寸的输入图像。内部使用 x//8, y//8 作为特征图尺寸。
    """
    def __init__(self, in_channels, out_channels, hidden_dim_channel, hidden_dim_token, x, y):
        super(Mixer_resize, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim_channel = hidden_dim_channel
        self.hidden_dim_token = hidden_dim_token
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)
        self.mix1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_channel, kernel_size=1),
            nn.LayerNorm([hidden_dim_channel, x//8, y//8]), # Layer Normalization
            nn.ReLU(),
            nn.Conv2d(hidden_dim_channel, out_channels, kernel_size=1)
        )

        self.mix2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_token, kernel_size=1),
            nn.LayerNorm([hidden_dim_token, x//8, y//8]), # Layer Normalization
            nn.ReLU(),
            nn.Conv2d(hidden_dim_token, out_channels, kernel_size=1)
        )
        self.gelu = nn.GELU()

    def forward(self, opt, sar):
        x1 = self.concat([opt, sar])
        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))
        channel_mixed = self.mix1(x1)
        x2 = opt + sar
        token_mixed = self.mix2(x2)
        mixed_features = self.gelu(x1 + x2 + channel_mixed + token_mixed)
        output = mixed_features + opt + sar
        return output


class Mixer_resize1(nn.Module):
    """
    GroupNorm版特征混合器 (GroupNorm Feature Mixer)

    与Mixer结构相同，但用GroupNorm(num_groups=1)替代LayerNorm，
    从而支持任意输入尺寸，不需要固定空间维度。
    """
    def __init__(self, in_channels, out_channels, hidden_dim_channel, hidden_dim_token):
        super(Mixer_resize1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim_channel = hidden_dim_channel
        self.hidden_dim_token = hidden_dim_token

        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)

        # 使用InstanceNorm或GroupNorm替代LayerNorm
        self.mix1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_channel, kernel_size=1),
            nn.GroupNorm(1, hidden_dim_channel),  # 使用GroupNorm替代LayerNorm
            nn.ReLU(),
            nn.Conv2d(hidden_dim_channel, out_channels, kernel_size=1)
        )

        self.mix2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_token, kernel_size=1),
            nn.GroupNorm(1, hidden_dim_token),  # 使用GroupNorm替代LayerNorm
            nn.ReLU(),
            nn.Conv2d(hidden_dim_token, out_channels, kernel_size=1)
        )
        self.gelu = nn.GELU()

    def forward(self, opt, sar):
        x1 = self.concat([opt, sar])
        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))
        channel_mixed = self.mix1(x1)
        x2 = opt + sar
        token_mixed = self.mix2(x2)
        mixed_features = self.gelu(x1 + x2 + channel_mixed + token_mixed)
        output = mixed_features + opt + sar
        return output
