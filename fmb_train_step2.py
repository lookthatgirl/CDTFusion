"""
FMB数据集 Step2 训练脚本 - 联合融合与分割训练 (Joint Fusion & Segmentation Training)

本脚本实现CDTFusion框架的第二阶段训练，使用FMB数据集。
目标: 冻结Step1预训练的编码器和迁移模块，训练融合和分割分支。

训练流程:
1. 加载Step1预训练的编码器和迁移模块（冻结，不计算梯度）
2. 提取跨域迁移后的特征 vi_allin 和 ir_allin
3. 分割分支: Mixer_seg → 分割解码器 → 分割结果
4. 融合分支: Mixer_fus → TaskInteraction(融合+分割交互) → 融合解码器 → 融合图像
5. 联合训练: 分割损失 + 融合损失

损失函数:
  seg_loss = Dice + 0.5 * CrossEntropy
  fus_loss = 1 - SSIM(target, fused)  (target = 0.5*Y + 0.5*IR)
  total_loss = seg_loss + fus_loss

训练模块: Mixer_seg, Mixer_fus, TaskInteraction, Decoder_seg(FMB), Decoder_fus
冻结模块: EncoderVi, EncoderIr, Transfer×2
数据集路径: datasets/fmb/train/{vi, ir, lbl}
模型保存: save/fmb/
"""

import torch
import torch.optim as optim
from util.seg_dataloader_fmb import get_train_loader
import torch.nn as nn
from util.encoder import EncoderVi
from util.encoder import EncoderIr
from util.res_decoder import head_seg_fmb
from util.res_decoder import head_fus
from util.transfer import transfer
from util.discriminator import DomDiscriminator
from util.mixer import Mixer
from util.taskinteraction import TaskInteraction
from util.loss_ssim import Fusionloss_ir
from util.dice_loss import dice_loss
from util.imageutil import RGB2YCrCb, YCrCb2RGB, RGB2YCrCb_Cuda1
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# ==================== 路径配置 ====================
dir_vi_train = "datasets/fmb/train/vi"    # 可见光训练图像
dir_ir_train = "datasets/fmb/train/ir"    # 红外训练图像
dir_seg_train = "datasets/fmb/train/lbl"  # 分割标签

# 模型保存路径
decoder_3_path = "save/fmb/decoder_3.pth"          # 分割解码器
decoder_4_path = "save/fmb/decoder_4.pth"          # 融合解码器
mixer_path = "save/fmb/mixer.pth"                  # 分割特征混合器
mixer_f_path = "save/fmb/mixer_f.pth"              # 融合特征混合器
task_interaction_path = "save/fmb/task_interaction.pth"  # 任务交互模块


def save_model(model, path):
    """保存模型权重到指定路径"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# ==================== 模型初始化 ====================
encoder_1 = EncoderVi()            # 可见光编码器（冻结）
encoder_2 = EncoderIr()            # 红外编码器（冻结）
transfer_vi_to_ir = transfer()     # VI→IR迁移（冻结）
transfer_ir_to_vi = transfer()     # IR→VI迁移（冻结）
domain_discriminator = DomDiscriminator()  # 域判别器（冻结）
decoder_3 = head_seg_fmb()         # FMB分割解码器（15类）
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
domain_discriminator.load_state_dict(torch.load('save/fmb/domain_discriminator.pth'))

encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
domain_discriminator.to(device)
decoder_3.to(device)
decoder_4.to(device)
mixer.to(device)
mixer_f.to(device)
taskinteraction.to(device)

# ==================== 超参数与优化器 ====================
batch_size = 4
epochs = 30
lr=0.0001
optimizer_decoder_3 = optim.Adam(decoder_3.parameters(), lr=lr)
optimizer_decoder_4 = optim.Adam(decoder_4.parameters(), lr=lr)
optimizer_mixer = optim.Adam(mixer.parameters(), lr=lr)
optimizer_mixer_f = optim.Adam(mixer_f.parameters(), lr=lr)
optimizer_taskinteraction = optim.Adam(taskinteraction.parameters(), lr=lr)

train_loader = get_train_loader(dir_vi_train, dir_ir_train, dir_seg_train, batch_size)

decoder_3.train()
decoder_4.train()
mixer.train()
mixer_f.train()
taskinteraction.train()

# ==================== 损失函数 ====================
loss = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()  # 分割交叉熵损失
fusionloss = Fusionloss_ir()      # 融合SSIM损失

# ==================== 训练循环 ====================
for epoch in range(epochs):
    for batch_idx, (vi_images, ir_images, seg_images) in enumerate(train_loader):
        vi_images, ir_images, seg_images = vi_images.to(device), ir_images.to(device), seg_images.to(device)
        # 将RGB转为YCrCb，提取Y通道（亮度）作为融合目标
        rgb_images_ycrcb = RGB2YCrCb_Cuda1(vi_images)
        opt_images = rgb_images_ycrcb[:, 0:1, :, :]  # Y通道 [B, 1, H, W]

        optimizer_decoder_3.zero_grad()
        optimizer_decoder_4.zero_grad()
        optimizer_mixer.zero_grad()
        optimizer_mixer_f.zero_grad()
        optimizer_taskinteraction.zero_grad()

        # 冻结的编码器和迁移模块提取特征（不计算梯度）
        with torch.no_grad():
            vi_features = encoder_1(vi_images)                       # [B, 512, H/8, W/8]
            ir_features = encoder_2(ir_images)                       # [B, 512, H/8, W/8]
            vi_allin = transfer_vi_to_ir(vi_features, ir_features)   # VI→IR迁移特征
            ir_allin = transfer_ir_to_vi(ir_features, vi_features)   # IR→VI迁移特征

        # ---- 分割分支 ----
        seg_features = mixer(vi_allin, ir_allin)     # 混合双模态特征用于分割
        seg_out = decoder_3(seg_features)            # [B, 15, H, W] FMB 15类分割输出

        seg_loss = dice_loss(seg_out, seg_images) + 0.5 * criterion(seg_out, seg_images)

        # ---- 融合分支 ----
        fus_features = mixer_f(vi_allin, ir_allin)                  # 混合双模态特征用于融合
        fus_features_taski = taskinteraction(fus_features, seg_features)  # 任务交互: 分割语义指导融合
        fus_out = decoder_4(fus_features_taski)                     # [B, 1, H, W] 融合输出

        fus_loss = fusionloss(opt_images, ir_images, fus_out)  # SSIM融合损失

        # 总损失 = 分割损失 + 融合损失
        total_loss = seg_loss + fus_loss
        total_loss.backward()

        optimizer_decoder_3.step()
        optimizer_mixer.step()
        optimizer_decoder_4.step()
        optimizer_mixer_f.step()
        optimizer_taskinteraction.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], total_loss: {total_loss.item():.4f}")

# ==================== 保存Step2训练的模型 ====================
save_model(decoder_3, decoder_3_path)
save_model(decoder_4, decoder_4_path)
save_model(mixer, mixer_path)
save_model(mixer_f, mixer_f_path)
save_model(taskinteraction, task_interaction_path)

