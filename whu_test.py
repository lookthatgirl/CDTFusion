"""
WHU数据集测试脚本 - 图像融合推理

与fmb_test.py结构相同，使用WHU卫星遥感数据集。
融合策略: 用融合网络生成的灰度图替换原始RGB图像的HSV-V通道。

模型权重: save/whu/
测试数据: datasets/whu/test/{vi, ir}
输出目录: fusion/whu_paper/
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
encoder_1.load_state_dict(torch.load('save/whu/encoder_1.pth'))
encoder_2.load_state_dict(torch.load('save/whu/encoder_2.pth'))
transfer_vi_to_ir.load_state_dict(torch.load('save/whu/transfer_vi_to_ir.pth'))
transfer_ir_to_vi.load_state_dict(torch.load('save/whu/transfer_ir_to_vi.pth'))

# 加载Step2训练权重
decoder_4.load_state_dict(torch.load('save/whu/decoder_4.pth'))
mixer.load_state_dict(torch.load('save/whu/mixer.pth'))
mixer_f.load_state_dict(torch.load('save/whu/mixer_f.pth'))
taskinteraction.load_state_dict(torch.load('save/whu/task_interaction.pth'))

# 将所有模型移到GPU
encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
decoder_4.to(device)
mixer.to(device)
mixer_f.to(device)
taskinteraction.to(device)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def fusion_images(rgb_path, ir_path, image_path):
    """
    融合单对VI/IR图像，用融合灰度替换HSV-V通道得到彩色融合图。
    流程同fmb_test.py的fusion_images。
    """
    rgb_image = Image.open(rgb_path).convert("RGB")
    ir_image = Image.open(ir_path).convert("L")

    img_array = np.array(rgb_image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)  # 转为HSV

    rgb_tensor = transform(rgb_image).unsqueeze(0).to(device)  # [1, 3, 512, 512]
    ir_tensor = transform(ir_image).unsqueeze(0).to(device)    # [1, 1, 512, 512]

    with torch.no_grad():
        # 编码 → 跨域迁移 → 特征混合 → 任务交互 → 融合解码
        test_vi_features = encoder_1(rgb_tensor)
        test_ir_features = encoder_2(ir_tensor)

        test_vi_allin = transfer_vi_to_ir(test_vi_features, test_ir_features)
        test_ir_allin = transfer_ir_to_vi(test_ir_features, test_vi_features)

        test_seg_features = mixer(test_vi_allin, test_ir_allin)
        test_fus_features = mixer_f(test_vi_allin, test_ir_allin)
        test_fus_features_taski = taskinteraction(test_fus_features, test_seg_features)

        generated_fus_images = decoder_4(test_fus_features_taski)
        # 裁剪到[0, 1]
        ones_1 = torch.ones_like(generated_fus_images)
        zeros_1 = torch.zeros_like(generated_fus_images)
        generated_fus_images = torch.where(generated_fus_images > ones_1, ones_1, generated_fus_images)
        generated_fus_images = torch.where(generated_fus_images < zeros_1, zeros_1, generated_fus_images)

        fus_img_pred = generated_fus_images.cpu().detach().numpy().squeeze()
        fus_img_pred = np.uint8(255.0 * fus_img_pred)

        # 替换V通道并转回RGB
        img_hsv[:, :, 2] = fus_img_pred
        modifited_rgb_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        modifited_rgb_img = Image.fromarray(modifited_rgb_img)
        modifited_rgb_img.save(image_path)

# ==================== 测试集融合 ====================
dir_vi_test = "datasets/whu/test/vi"
dir_ir_test = "datasets/whu/test/ir"
output_dir = 'fusion/whu_paper'
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
