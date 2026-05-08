"""
数据加载工具模块 (Data Loading Utilities)

本文件实现了Step1训练用的数据加载器，用于加载可见光(VI)/红外(IR)图像对。

包含:
- extract_number: 从文件名中提取数字用于排序
- CustomDataset: 自定义数据集类，加载RGB+IR图像对
- get_train_loader: 创建训练集DataLoader
- get_test_loader: 创建测试集DataLoader

注意: 此加载器不包含分割标签，仅用于Step1的跨域迁移预训练。
"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset


def extract_number(filename):
    """从文件名中提取数字，用于按数字顺序排序文件列表"""
    return int(''.join(filter(str.isdigit, filename)))


class CustomDataset(Dataset):
    """
    自定义图像对数据集 (Custom Image Pair Dataset)

    加载配对的可见光RGB图像和红外灰度图像。
    图像按文件名中的数字排序以确保一一对应。

    参数:
        root_dir_rgb: 可见光图像目录
        root_dir_ir: 红外图像目录
        transform: 图像变换
    """
    def __init__(self, root_dir_rgb, root_dir_ir, transform=None):
        self.root_dir_rgb = root_dir_rgb
        self.root_dir_ir = root_dir_ir
        self.transform = transform
        self.rgb_images = sorted(os.listdir(root_dir_rgb), key=extract_number)
        self.ir_images = sorted(os.listdir(root_dir_ir), key=extract_number)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image_name = os.path.join(self.root_dir_rgb, self.rgb_images[idx])
        ir_image_name = os.path.join(self.root_dir_ir, self.ir_images[idx])

        rgb_image = Image.open(rgb_image_name).convert("RGB")  # 加载为3通道RGB
        ir_image = Image.open(ir_image_name).convert("L")       # 加载为1通道灰度

        if self.transform:
            rgb_image = self.transform(rgb_image)  # [3, H, W]
            ir_image = self.transform(ir_image)    # [1, H, W]

        return rgb_image, ir_image

def get_train_loader(dir_rgb_train, dir_ir_train, batch_size):
    """创建训练集DataLoader，仅包含ToTensor变换（无Resize）"""
    transform = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(root_dir_rgb=dir_rgb_train, root_dir_ir=dir_ir_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_test_loader(dir_rgb_test, dir_ir_test, batch_size):
    """创建测试集DataLoader，仅包含ToTensor变换（无Resize），不打乱顺序"""
    transform = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDataset(root_dir_rgb=dir_rgb_test, root_dir_ir=dir_ir_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


