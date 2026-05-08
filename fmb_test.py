"""
FMB数据集测试脚本 - 图像融合推理 (Image Fusion Inference)

本脚本加载训练好的模型权重，对FMB测试集的图像对进行融合。

融合流程:
1. 读取可见光(RGB)和红外(IR)图像对
2. 将可见光图像转为HSV色彩空间
3. 编码器提取特征 → 跨域迁移 → 特征混合 → 任务交互 → 融合解码器
4. 融合输出裁剪到[0,1]并转为灰度图
5. 用融合灰度图替换HSV的V通道（亮度）
6. 转换回RGB，得到彩色融合图像

模型权重: save/fmb/
测试数据: datasets/fmb/test/{vi, ir}
输出目录: fusion/fmb/
"""

import torch
from util.encoder import EncoderVi
from util.encoder import EncoderIr
from util.res_decoder import head_fus
from util.transfer import transfer
from util.mixer import Mixer
from util.taskinteraction import TaskInteraction
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== 模型初始化与权重加载 ====================
encoder_1 = EncoderVi()            # 可见光编码器
encoder_2 = EncoderIr()            # 红外编码器
transfer_vi_to_ir = transfer()     # VI→IR跨域迁移
transfer_ir_to_vi = transfer()     # IR→VI跨域迁移
decoder_4 = head_fus()             # 融合解码器
hidden_dim_channel = 1024
hidden_dim_token = 512
mixer = Mixer(512, 512, hidden_dim_channel, hidden_dim_token)      # 分割特征混合器
mixer_f = Mixer(512, 512, hidden_dim_channel, hidden_dim_token)    # 融合特征混合器
taskinteraction = TaskInteraction(in_channels=512)                 # 任务交互模块

# 加载Step1预训练权重
encoder_1.load_state_dict(torch.load('save/fmb/encoder_1.pth'))
encoder_2.load_state_dict(torch.load('save/fmb/encoder_2.pth'))
transfer_vi_to_ir.load_state_dict(torch.load('save/fmb/transfer_vi_to_ir.pth'))
transfer_ir_to_vi.load_state_dict(torch.load('save/fmb/transfer_ir_to_vi.pth'))

# 加载Step2训练权重
decoder_4.load_state_dict(torch.load('save/fmb/decoder_4.pth'))
mixer.load_state_dict(torch.load('save/fmb/mixer.pth'))
mixer_f.load_state_dict(torch.load('save/fmb/mixer_f.pth'))
taskinteraction.load_state_dict(torch.load('save/fmb/task_interaction.pth'))

# 将所有模型移到GPU
encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
decoder_4.to(device)
mixer.to(device)
mixer_f.to(device)
taskinteraction.to(device)

# 图像预处理: Resize到512x512并转为张量
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def fusion_images(rgb_path, ir_path, image_path):
    """
    融合单对可见光/红外图像并保存结果

    融合策略: 用网络生成的融合灰度图替换原始RGB图像的HSV-V通道，
    保留原始的色调(H)和饱和度(S)信息，从而得到彩色融合图像。

    参数:
        rgb_path: 可见光RGB图像路径
        ir_path: 红外灰度图像路径
        image_path: 融合结果保存路径
    """
    rgb_image = Image.open(rgb_path).convert("RGB")
    ir_image = Image.open(ir_path).convert("L")

    # 将原始RGB转换为HSV，用于后续替换V通道
    img_array = np.array(rgb_image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # 预处理图像为模型输入
    rgb_tensor = transform(rgb_image).unsqueeze(0).to(device)  # [1, 3, 512, 512]
    ir_tensor = transform(ir_image).unsqueeze(0).to(device)    # [1, 1, 512, 512]

    # 前向推理（不计算梯度）
    with torch.no_grad():
        # 编码器提取特征
        test_vi_features = encoder_1(rgb_tensor)   # [1, 512, H/8, W/8]
        test_ir_features = encoder_2(ir_tensor)    # [1, 512, H/8, W/8]

        # 跨域迁移
        test_vi_allin = transfer_vi_to_ir(test_vi_features, test_ir_features)
        test_ir_allin = transfer_ir_to_vi(test_ir_features, test_vi_features)

        # 特征混合 + 任务交互
        test_seg_features = mixer(test_vi_allin, test_ir_allin)        # 分割分支特征
        test_fus_features = mixer_f(test_vi_allin, test_ir_allin)      # 融合分支特征
        test_fus_features_taski = taskinteraction(test_fus_features, test_seg_features)  # 任务交互

        # 融合解码器生成融合图像
        generated_fus_images = decoder_4(test_fus_features_taski)  # [1, 1, H, W]

        # 裁剪到[0, 1]范围
        ones_1 = torch.ones_like(generated_fus_images)
        zeros_1 = torch.zeros_like(generated_fus_images)
        generated_fus_images = torch.where(generated_fus_images > ones_1, ones_1, generated_fus_images)
        generated_fus_images = torch.where(generated_fus_images < zeros_1, zeros_1, generated_fus_images)

        # 转换为uint8灰度图 [0, 255]
        fus_img_pred = generated_fus_images.cpu().detach().numpy().squeeze()
        fus_img_pred = np.uint8(255.0 * fus_img_pred)

        # 用融合灰度图替换HSV的V通道，保留原始色彩信息
        img_hsv[:, :, 2] = fus_img_pred
        modifited_rgb_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        modifited_rgb_img = Image.fromarray(modifited_rgb_img)
        modifited_rgb_img.save(image_path)


# ==================== 测试集融合 ====================
dir_vi_test = "datasets/fmb/test/vi"
dir_ir_test = "datasets/fmb/test/ir"
output_dir = 'fusion/fmb'
os.makedirs(output_dir, exist_ok=True)

# 遍历测试集所有图像进行融合
test_vi_files = [f for f in os.listdir(dir_vi_test) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
for filename in test_vi_files:
    rgb_path = os.path.join(dir_vi_test, filename)
    ir_path = os.path.join(dir_ir_test, filename)
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(ir_path):
        print(f"Skipping {filename}: IR image not found")
        continue

    fusion_images(rgb_path, ir_path, output_path)
    print(f"Processed test image: {filename}")
