"""
局部均值-方差损失模块 (Local Mean-Variance Loss Module)

本文件实现了局部均值-方差匹配损失，用于约束两个域迁移后的特征
具有相似的局部统计特性（均值和方差）。

用于Step1训练，确保VI->IR和IR->VI迁移后的特征分布一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalMeanVarianceLoss(nn.Module):
    """
    局部均值-方差损失 (Local Mean-Variance Loss)

    通过平均池化计算局部窗口内的均值和方差，
    然后计算两个特征图的均值差和方差差的MSE损失。

    目的: 使两个迁移后的特征在局部窗口内具有相似的均值和方差，
    促进两个域的特征分布对齐。

    参数:
        window_size: 局部窗口大小
    """
    def __init__(self, window_size):
        super(LocalMeanVarianceLoss, self).__init__()
        self.window_size = window_size

    def forward(self, feature1, feature2):
        """
        参数:
            feature1: 第一个迁移特征 [B, C, H, W]
            feature2: 第二个迁移特征 [B, C, H, W]

        返回:
            mean_loss: 局部均值差的MSE损失
            variance_loss: 局部方差差的MSE损失
        """
        # 计算局部均值: 通过平均池化实现
        mean1 = F.avg_pool2d(feature1, self.window_size, stride=1, padding=self.window_size // 2)
        mean2 = F.avg_pool2d(feature2, self.window_size, stride=1, padding=self.window_size // 2)
        # 计算局部方差: E[X^2] - (E[X])^2
        variance1 = F.avg_pool2d(feature1**2, self.window_size, stride=1, padding=self.window_size // 2) - mean1**2
        variance2 = F.avg_pool2d(feature2**2, self.window_size, stride=1, padding=self.window_size // 2) - mean2**2

        # 均值匹配损失: 两个特征的局部均值应该接近
        mean_loss = torch.mean((mean1 - mean2)**2)
        # 方差匹配损失: 两个特征的局部方差应该接近
        variance_loss = torch.mean((variance1 - variance2)**2)

        return mean_loss, variance_loss
