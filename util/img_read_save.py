"""
图像读写工具模块 (Image I/O Utilities)

本文件提供图像的读取和保存功能。

包含:
- image_read_cv2: 使用OpenCV读取图像，支持RGB/GRAY/YCrCb模式
- img_save: 使用skimage保存图像为PNG格式
"""

import numpy as np
import cv2
import os
from skimage.io import imsave


def image_read_cv2(path, mode='RGB'):
    """
    使用OpenCV读取图像

    参数:
        path: 图像文件路径
        mode: 色彩模式，支持 'RGB' / 'GRAY' / 'YCrCb'

    返回:
        float32类型的numpy数组
    """
    img_BGR = cv2.imread(path).astype('float32')  # OpenCV默认读取为BGR格式
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image, imagename, savepath):
    """
    保存图像为PNG格式

    参数:
        image: 要保存的图像数组
        imagename: 图像文件名（不含扩展名）
        savepath: 保存目录，不存在则自动创建
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imsave(os.path.join(savepath, "{}.png".format(imagename)), image)
