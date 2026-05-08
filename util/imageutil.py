"""
色彩空间转换工具模块 (Color Space Conversion Utilities)

本文件实现了RGB和YCrCb色彩空间之间的转换。

在测试阶段，融合图像的生成过程为:
1. 将可见光RGB图像转换为YCrCb
2. 用融合网络生成的灰度图替换Y通道（亮度）
3. 保留原始的Cr和Cb通道（色度信息）
4. 将修改后的YCrCb转换回RGB，得到彩色融合图像

包含:
- RGB2YCrCb: RGB转YCrCb (cuda:0)
- YCrCb2RGB: YCrCb转RGB (cuda:0)
- RGB2YCrCb_Cuda1: RGB转YCrCb (cuda:1)
"""

import torch


def RGB2YCrCb(input_im):
    """
    RGB转YCrCb色彩空间 (cuda:0)

    输入: [B, 3, H, W] RGB图像张量
    输出: [B, 3, H, W] YCrCb图像张量

    转换公式:
        Y  = 0.299R + 0.587G + 0.114B   (亮度)
        Cr = (R - Y) * 0.713 + 0.5      (红色色度)
        Cb = (B - Y) * 0.564 + 0.5      (蓝色色度)
    """
    device = torch.device("cuda:0")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # 展平为(N, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    # 计算YCrCb各通道
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    # 恢复为(B, 3, H, W)形状
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    """
    YCrCb转RGB色彩空间 (cuda:0)

    输入: [B, 3, H, W] YCrCb图像张量
    输出: [B, 3, H, W] RGB图像张量

    通过矩阵乘法进行线性变换，将YCrCb各通道转换回RGB。
    """
    device = torch.device("cuda:0")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # 展平为(N, 3)
    # YCrCb -> RGB 转换矩阵
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)  # 矩阵乘法转换
    # 恢复为(B, 3, H, W)形状
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb_Cuda1(input_im):
    """
    RGB转YCrCb色彩空间 (cuda:1)

    与RGB2YCrCb功能相同，但在cuda:1设备上运行。
    用于多 GPU 训练场景。
    """
    device = torch.device("cuda:1")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
