"""
任务交互模块 (Task Interaction Module)

本文件实现了融合任务和分割任务之间的信息交互机制。

包含:
- TaskInteraction: 基于注意力的任务交互模块，让分割语义信息指导融合过程
- CatAndConvolve: 简单的拼接+卷积融合模块（已定义但未使用）

核心思想: 利用分割任务提取的语义特征来增强融合特征，
使得融合结果更好地保留语义相关的结构信息。
"""

import torch
import torch.nn as nn


class TaskInteraction(nn.Module):
    """
    任务交互模块 (Task Interaction)

    将融合特征和分割特征进行交互，让分割语义信息指导融合过程。

    工作流程:
    1. 拼接融合特征和分割特征 -> 1x1卷积降维 + 残差连接
    2. 自注意力 (MultiheadAttention) 全局建模
    3. MLP 进一步变换
    4. Q-K-V 交叉注意力: 融合特征生成Q，处理后的特征生成K、V
    5. 输出 = 融合特征 + 注意力加权结果

    输入: x1 (融合特征) [B, C, H, W], x2 (分割特征) [B, C, H, W]
    输出: 增强后的融合特征 [B, C, H, W]

    参数:
        in_channels: 输入特征通道数
    """
    def __init__(self, in_channels):
        super(TaskInteraction, self).__init__()

        # 特征拼接后的降维卷积: 2C -> C
        self.concat = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.bn = nn.BatchNorm2d(in_channels)

        # 自注意力模块: 全局建模空间位置间的关系
        self.self_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1)

        # MLP: 进一步变换注意力输出
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.in_channels = in_channels

        self.bn_x1 = nn.BatchNorm2d(in_channels)
        self.bn_x2 = nn.BatchNorm2d(in_channels)

        # Q-K-V 交叉注意力的线性投影
        self.linear_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Query: 来自融合特征
        self.linear_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Key: 来自处理后的特征
        self.linear_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Value: 来自处理后的特征

    def forward(self, x1, x2):
        """
        前向传播

        参数:
            x1: 融合特征 [B, C, H, W]
            x2: 分割特征 [B, C, H, W]

        返回:
            任务交互增强后的融合特征 [B, C, H, W]
        """
        # 步骤1: 拼接 + 1x1卷积降维 + 残差连接
        x = torch.cat((x1, x2), dim=1)    # [B, 2C, H, W]
        x = x1 + self.concat(x)            # [B, C, H, W] 残差连接

        x = self.bn(x)

        # 步骤2: 自注意力 - 将空间维度展平为序列
        x = x.view(x.shape[0], x.shape[1], -1).permute(2, 0, 1)  # [HW, B, C]
        x, _ = self.self_attention(x, x, x)  # 自注意力
        x = x.permute(1, 2, 0).view(x1.shape)  # 恢复为 [B, C, H, W]

        # 步骤3: MLP变换
        x = self.mlp(x)
        x1 = self.bn_x1(x1)    # 对融合特征做归一化
        x2 = self.bn_x2(x)     # 对处理后的特征做归一化

        # 步骤4: Q-K-V 交叉注意力
        q = self.linear_q(x1)   # Query 来自融合特征
        k = self.linear_k(x2)   # Key 来自处理后的特征
        v = self.linear_v(x2)   # Value 来自处理后的特征

        # 展平空间维度以计算注意力
        q = q.view(q.size(0), q.size(1), -1)   # [B, C, HW]
        k = k.view(k.size(0), k.size(1), -1)   # [B, C, HW]
        v = v.view(v.size(0), v.size(1), -1)   # [B, C, HW]

        # 计算注意力分数: Q^T * K
        attention_scores = torch.bmm(q.transpose(1, 2), k)  # [B, HW, HW]

        # 软化为注意力权重
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=2)

        # 加权求和
        weighted_sum = torch.bmm(attention_weights, v.transpose(1, 2))  # [B, HW, C]

        # 恢复空间维度
        weighted_sum = weighted_sum.view(weighted_sum.size(0), self.in_channels, x2.size(2), x2.size(3))

        # 步骤5: 残差连接 - 融合特征 + 注意力输出
        output = x1 + weighted_sum

        return output



class CatAndConvolve(nn.Module):
    """
    拼接+卷积融合模块 (Concatenate and Convolve)

    简单地将两个512通道特征拼接为1024通道，再通过3x3卷积降维到512通道。
    （已定义但未在当前训练流程中使用）
    """
    def __init__(self):
        super(CatAndConvolve, self).__init__()
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)

    def forward(self, tensor1, tensor2):
        # 拼接两个特征: [B, 512, H, W] + [B, 512, H, W] -> [B, 1024, H, W]
        concatenated_tensor = torch.cat((tensor1, tensor2), dim=1)
        # 卷积降维: [B, 1024, H, W] -> [B, 512, H, W]
        output_tensor = self.conv(concatenated_tensor)

        return output_tensor
