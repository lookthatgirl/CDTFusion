"""
FMB数据集加载器 (FMB Dataset Loader)

本文件实现了FMB数据集的数据加载，用于Step2的联合融合+分割训练。
加载三元组: 可见光图像(RGB) + 红外图像(IR) + 分割标签(Mask)。

FMB数据集特点:
- 分割标签为灰度图，像素值直接代表类别索引 (0-14)
- 共 15 个类别（包含背景 + 14个语义类别）
- 图像统一Resize到51 2x512
"""

import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import numpy as np
import torch


def extract_number(filename):
    """从文件名中提取数字，用于按数字顺序排序文件列表"""
    return int(''.join(filter(str.isdigit, filename)))


class CustomDataset(Dataset):
    """
    FMB数据集自定义加载器

    加载配对的RGB图像、IR图像和分割标签。
    分割标签为灰度图，像素值直接代表类别索引。

    参数:
        root_dir_rgb: 可见光图像目录
        root_dir_ir: 红外图像目录
        root_dir_seg: 分割标签目录
        transform: 图像变换
    """
    def __init__(self, root_dir_rgb, root_dir_ir, root_dir_seg, transform=None):
        self.root_dir_rgb = root_dir_rgb
        self.root_dir_ir = root_dir_ir
        self.root_dir_seg = root_dir_seg
        self.transform = transform
        self.rgb_images = sorted(os.listdir(root_dir_rgb), key=extract_number)
        self.ir_images = sorted(os.listdir(root_dir_ir), key=extract_number)
        self.seg_images = sorted(os.listdir(root_dir_seg), key=extract_number)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image_name = os.path.join(self.root_dir_rgb, self.rgb_images[idx])
        ir_image_name = os.path.join(self.root_dir_ir, self.ir_images[idx])
        seg_image_name = os.path.join(self.root_dir_seg, self.seg_images[idx])

        rgb_image = Image.open(rgb_image_name).convert("RGB")  # 加载为3通道RGB
        ir_image = Image.open(ir_image_name).convert("L")       # 加载为1通道灰度
        seg_image = Image.open(seg_image_name).convert("L")     # 加载灰度标签

        seg_image = np.array(seg_image)  # 像素值即类别索引 (0-14)

        if self.transform:
            rgb_image = self.transform(rgb_image)  # [3, 512, 512]
            ir_image = self.transform(ir_image)    # [1, 512, 512]
            mask = torch.from_numpy(seg_image).long()  # [H, W] 注意: 未进Resize

        return rgb_image, ir_image, mask

def get_train_loader(dir_rgb_train, dir_ir_train, dir_seg_train, batch_size):
    """创建FMB训练集DataLoader，图像Resize到512x512"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(root_dir_rgb=dir_rgb_train, root_dir_ir=dir_ir_train, root_dir_seg=dir_seg_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def get_test_loader(dir_rgb_test, dir_ir_test, dir_seg_test, batch_size):
    """创建FMB测试集DataLoader，图像Resize到512x512，不打乱顺序"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDataset(root_dir_rgb=dir_rgb_test, root_dir_ir=dir_ir_test, root_dir_seg=dir_seg_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
