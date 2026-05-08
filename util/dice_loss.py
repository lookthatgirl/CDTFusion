"""
Dice损失函数模块 (Dice Loss Module)

实现了用于语义分割任务的Dice损失函数。
Dice损失通过衡量预测和真实标签之间的重叠程度来评估分割质量，
对类别不平衡的情况特别有效。
"""

import torch


def dice_loss(predicted, target, num_classes=15):
    """
    计算多类别Dice损失

    对每个类别分别计算Dice系数，然后取所有类别的平均值。
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - mean(Dice)

    参数:
        predicted: 预测输出 [B, num_classes, H, W]，每个类别的概率图
        target: 真实标签 [B, H, W]，每个像素的类别索引
        num_classes: 类别数，默认15（FMB数据集）

    返回:
        1 - 平均Dice系数，范围[0, 1]，越小表示分割越好
    """
    dice_scores = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        predicted_class = predicted[:, class_idx, ...]     # 第class_idx类的预测概率图
        target_class = (target == class_idx).float()        # 真实标签的二值化mask
        intersection = torch.sum(predicted_class * target_class)  # 交集
        union = torch.sum(predicted_class) + torch.sum(target_class)  # 并集
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)  # +1e-5防止除零
        dice_scores[class_idx] = dice
    return 1.0 - dice_scores.mean()  # 返回 1 - 平均Dice作为损失
