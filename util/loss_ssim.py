"""
SSIM损失函数模块 (SSIM Loss Module)

本文件实现了基于结构相似性(SSIM)的损失函数，用于图像融合质量评估。

包含:
- gaussian: 生成一维高斯核
- create_window: 创建二维高斯窗口
- _ssim: SSIM计算核心函数（支持beta参数缩放）
- SSIMLoss: SSIM损失类（主要使用）
- Fusionloss_ir: 融合损失类，计算融合图与目标图的SSIM损失
- Contrast: 对比度计算函数（未使用）
- ssim: 独立的SSIM计算函数（未使用）
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    """生成一维高斯分布向量，用于构建高斯模糊窗口"""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    """
    创建二维高斯窗口

    通过外积将一维高斯核扩展为二维，然后复制到每个通道。
    用于SSIM计算中的均值和方差估计。
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, beta, window, window_size, channel, size_average=True):
    """
    计算两张图像的SSIM值

    注意: 该实现对sigma1_sq应用了beta**2缩放，对sigma12应用了beta缩放，
    但sigma2_sq未缩放。这是一种非对称SSIM变体，
    用于强调某一个模态的结构信息。

    参数:
        img1, img2: 输入图像 [B, C, H, W]
        beta: 方差缩放系数
        window: 高斯模糊窗口
        window_size: 窗口大小
        channel: 通道数
        size_average: 是否对所有像素取平均

    返回:
        SSIM值，范围[-1, 1]，值越大表示越相似
    """
    # 计算局部均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    # 计算局部方差和协方差（带beta缩放）
    sigma1_sq =(beta**2)*(F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq)
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = beta*(F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2)

    # SSIM公式中的稳定性常数
    C1 = 0.01**2
    C2 = 0.03**2

    # SSIM公式: (2*mu1*mu2 + C1)(2*sigma12 + C2) / (mu1^2 + mu2^2 + C1)(sigma1^2 + sigma2^2 + C2)
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def Contrast(img1, img2, window_size=7, channel=1):
    """计算两张图像的局部方差（对比度），未在当前训练流程中使用"""
    window = create_window(window_size, channel)    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq

    
class SSIMLoss(torch.nn.Module):
    """
    SSIM损失函数类

    封装_ssim函数为nn.Module，缓存高斯窗口以提高效率。
    支持beta参数用于非对称缩放。

    参数:
        window_size: 高斯窗口大小，默认7
        size_average: 是否对所有像素取平均
    """
    def __init__(self, window_size=7, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, beta):
        """
        计算SSIM损失

        参数:
            img1, img2: 输入图像 [B, C, H, W]
            beta: 方差缩放系数，控制对img1的结构信息强调程度
        """
        (_, channel, _, _) = img1.size()
        # 缓存高斯窗口，避免重复创建
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, beta, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=7, size_average=True):
    """独立SSIM计算函数，未在当前训练流程中使用"""
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class Fusionloss_ir(torch.nn.Module):
    """
    融合损失函数 (Fusion Loss)

    计算融合图像与目标图像之间的SSIM损失。
    目标图像 = 0.5 * 红外图像 + 0.5 * 可见光图像（均等混合）
    损失 = 1 - SSIM(目标, 融合图, beta=4.5)

    用于Step2训练中的融合分支。
    """
    def __init__(self):
        super(Fusionloss_ir, self).__init__()
        self.ssim_loss=SSIMLoss(window_size=7)

    def forward(self, image_vis, image_ir, generate_img):
        # 目标图像: 红外和可见光的均等混合
        target=0.5*image_ir +0.5*image_vis
        # 损失 = 1 - SSIM，SSIM越大损失越小
        loss_total = 1-self.ssim_loss(target, generate_img, 4.5)

        return loss_total
