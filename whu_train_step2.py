"""
WHU数据集 Step2 训练脚本 - 联合融合与分割训练

与fmb_train_step2.py结构相同，使用WHU卫星遥感数据集（8类分割）。
目标: 冻结Step1预训练模块，联合训练融合和分割分支。

差异:
- 使用head_seg_whu分割解码器（输出8类）
- dice_loss指定num_classes=8
- 使用RGB2YCrCb（cuda:0）而非RGB2YCrCb_Cuda1

数据集路径: datasets/whu/train/{vi, ir, lbl}
模型保存: save/whu/
"""

import torch
import torch.optim as optim
from util.seg_dataloader_whu import get_train_loader
import torch.nn as nn
from util.encoder import EncoderVi
from util.encoder import EncoderIr
from util.res_decoder import head_seg_whu
from util.res_decoder import head_fus
from util.transfer import transfer
from util.mixer import Mixer
from util.taskinteraction import TaskInteraction
from util.loss_ssim import Fusionloss_ir
from util.dice_loss import dice_loss
from util.imageutil import RGB2YCrCb,YCrCb2RGB
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==================== 路径配置 ====================
dir_vi_train = "datasets/whu/train/vi"
dir_ir_train = "datasets/whu/train/ir"
dir_seg_train = "datasets/whu/train/lbl"

decoder_3_path = "save/whu/decoder_3.pth"
decoder_4_path = "save/whu/decoder_4.pth"
mixer_path = "save/whu/mixer.pth"
mixer_f_path = "save/whu/mixer_f.pth"
task_interaction_path = "save/whu/task_interaction.pth"


def save_model(model, path):
    """保存模型权重到指定路径"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# ==================== 模型初始化 ====================
encoder_1 = EncoderVi()            # 可见光编码器（冻结）
encoder_2 = EncoderIr()            # 红外编码器（冻结）
transfer_vi_to_ir = transfer()     # VI→IR迁移（冻结）
transfer_ir_to_vi = transfer()     # IR→VI迁移（冻结）
decoder_3 = head_seg_whu()         # WHU分割解码器（8类）
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


encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
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
criterion = nn.CrossEntropyLoss()
fusionloss = Fusionloss_ir()

# ==================== 训练循环 ====================
for epoch in range(epochs):
    for batch_idx, (vi_images, ir_images, seg_images) in enumerate(train_loader):
        vi_images, ir_images, seg_images = vi_images.to(device), ir_images.to(device), seg_images.to(device)
        # 提取Y通道作为融合目标
        rgb_images_ycrcb = RGB2YCrCb(vi_images)
        opt_images = rgb_images_ycrcb[:, 0:1, :, :]  # Y通道

        optimizer_decoder_3.zero_grad()
        optimizer_decoder_4.zero_grad()
        optimizer_mixer.zero_grad()
        optimizer_mixer_f.zero_grad()
        optimizer_taskinteraction.zero_grad()

        # 冻结模块提取迁移特征
        with torch.no_grad():
            vi_features = encoder_1(vi_images)
            ir_features = encoder_2(ir_images)
            vi_allin = transfer_vi_to_ir(vi_features, ir_features)
            ir_allin = transfer_ir_to_vi(ir_features, vi_features)

        # ---- 分割分支 ----
        seg_features = mixer(vi_allin, ir_allin)
        seg_out = decoder_3(seg_features)  # [B, 8, H, W]

        seg_loss = dice_loss(seg_out, seg_images, num_classes=8) + 0.5 * criterion(seg_out, seg_images)

        # ---- 融合分支 ----
        fus_features = mixer_f(vi_allin, ir_allin)
        fus_features_taski = taskinteraction(fus_features, seg_features)
        fus_out = decoder_4(fus_features_taski)  # [B, 1, H, W]

        fus_loss = fusionloss(opt_images, ir_images, fus_out)

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
