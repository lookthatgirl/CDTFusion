"""
FMB数据集 Step1 训练脚本 - 跨域迁移预训练 (Cross-Domain Transfer Pre-training)

本脚本实现CDTFusion框架的第一阶段训练，使用FMB数据集。
目标: 训练编码器和跨域迁移模块，学习可见光与红外模态之间的特征迁移。

训练流程:
1. 对抗训练阶段:
   - 冻结编码器，训练域判别器区分VI/IR特征
   - 判别器学习区分可见光域和红外域的特征
2. 生成训练阶段:
   - 冻结判别器，训练编码器+迁移模块+重建解码器
   - VI→IR 跨域迁移: 将VI特征迁移到IR域，重建IR图像
   - IR→VI 跨域迁移: 将IR特征迁移到VI域，重建VI图像
   - 域对抗损失: 让迁移后特征欺骗判别器（域不变）
   - 均值-方差对齐: 约束两个迁移分支的特征分布一致

损失函数:
  total_loss = L1重建 + SSIM重建 + λ_domain * 域对抗 + λ_mv * 均值方差对齐

训练模块: EncoderVi, EncoderIr, Transfer×2, DomDiscriminator, Decoder_1, Decoder_2
数据集路径: datasets/fmb/train/{vi, ir}
模型保存: save/fmb/
"""

import torch
import torch.optim as optim
import kornia.losses as losses
from util.data_utils import get_train_loader
import torch.nn.functional as F
import torch.nn as nn
from util.encoder import EncoderVi
from util.encoder import EncoderIr
from util.res_decoder import head_1
from util.res_decoder import head_2
from util.transfer import transfer
from util.discriminator import DomDiscriminator
from util.MeanVarianceLoss import LocalMeanVarianceLoss
import warnings
warnings.filterwarnings("ignore")

# ==================== 超参数与路径配置 ====================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size = 4   # 批大小
epochs = 20      # 训练轮数
lr = 0.0001      # 学习率

# 训练数据路径
dir_vi_train = "datasets/fmb/train/vi"
dir_ir_train = "datasets/fmb/train/ir"

# 模型保存路径
encoder_1_path = "save/fmb/encoder_1.pth"
encoder_2_path = "save/fmb/encoder_2.pth"
transfer_vi_to_ir_path = "save/fmb/transfer_vi_to_ir.pth"
transfer_ir_to_vi_path = "save/fmb/transfer_ir_to_vi.pth"
domain_discriminator_path = "save/fmb/domain_discriminator.pth"
decoder_1_path = "save/fmb/decoder_1.pth"
decoder_2_path = "save/fmb/decoder_2.pth"


def save_model(model, path):
    """保存模型权重到指定路径"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

train_loader = get_train_loader(dir_vi_train, dir_ir_train, batch_size)

# ==================== 模型初始化 ====================
encoder_1 = EncoderVi()            # 可见光编码器
encoder_2 = EncoderIr()            # 红外编码器
transfer_vi_to_ir = transfer()     # VI→IR 跨域迁移模块
transfer_ir_to_vi = transfer()     # IR→VI 跨域迁移模块
domain_discriminator = DomDiscriminator()  # 域判别器
decoder_1 = head_1()               # VI重建解码器（3通道输出）
decoder_2 = head_2()               # IR重建解码器（1通道输出）

# 将所有模型移到GPU
encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
domain_discriminator.to(device)
decoder_1.to(device)
decoder_2.to(device)

# ==================== 优化器 ====================
optimizer_encoder_1 = optim.Adam(encoder_1.parameters(), lr=lr)
optimizer_encoder_2 = optim.Adam(encoder_2.parameters(), lr=lr)
optimizer_transfer_vi_to_ir = optim.Adam(transfer_vi_to_ir.parameters(), lr=lr)
optimizer_transfer_ir_to_vi = optim.Adam(transfer_ir_to_vi.parameters(), lr=lr)
optimizer_domain_discriminator = optim.Adam(domain_discriminator.parameters(), lr=lr)
optimizer_decoder_1 = optim.Adam(decoder_1.parameters(), lr=lr)
optimizer_decoder_2 = optim.Adam(decoder_2.parameters(), lr=lr)

encoder_1.train()
encoder_2.train()
transfer_vi_to_ir.train()
transfer_ir_to_vi.train()
domain_discriminator.train()
decoder_1.train()
decoder_2.train()

# ==================== 损失函数 ====================
ssim_loss = losses.SSIMLoss(window_size=11, reduction='mean')  # 结构相似性损失
bce_loss = nn.BCELoss()                                        # 二元交叉熵（域判别）
loss_calculator = LocalMeanVarianceLoss(window_size=4)          # 局部均值-方差对齐损失

# ==================== 训练循环 ====================
for epoch in range(epochs):
    lambda_domain = 0.5       # 域对抗损失权重
    lambda_mean_vari = 0.5    # 均值-方差损失权重
    for batch_idx, (vi_images, ir_images) in enumerate(train_loader):
        vi_images, ir_images = vi_images.to(device), ir_images.to(device)

        # ---- 阶段1: 训练域判别器 ----
        # 冻结编码器，提取特征（不计算梯度）
        encoder_1.eval()
        encoder_2.eval()
        with torch.no_grad():
            vi_features = encoder_1(vi_images)   # [B, 512, H/8, W/8]
            ir_features = encoder_2(ir_images)   # [B, 512, H/8, W/8]

        # 域标签: VI=1, IR=0, 混合=0.5
        domain_vi_labels = torch.ones(vi_features.size(0), 1).to(device)
        domain_ir_labels = torch.zeros(ir_features.size(0), 1).to(device)
        domain_un_labels = (domain_ir_labels + domain_vi_labels) / 2

        # 训练判别器区分VI和IR特征
        domain_discriminator.train()
        for param in domain_discriminator.parameters():
            param.requires_grad = True

        optimizer_domain_discriminator.zero_grad()

        domain_vi_pred = domain_discriminator(vi_features)  # 预测VI特征的域标签
        domain_ir_pred = domain_discriminator(ir_features)  # 预测IR特征的域标签

        # 判别器损失: 让判别器正确区分VI和IR
        domain_loss_real = bce_loss(domain_vi_pred, domain_vi_labels) + bce_loss(domain_ir_pred, domain_ir_labels)
        domain_loss_real.backward()
        optimizer_domain_discriminator.step()

        # ---- 阶段2: 训练编码器+迁移模块+解码器 ----
        # 冻结判别器，训练其余模块
        encoder_1.train()
        encoder_2.train()
        domain_discriminator.eval()
        for param in domain_discriminator.parameters():
            param.requires_grad = False

        optimizer_encoder_1.zero_grad()
        optimizer_encoder_2.zero_grad()
        optimizer_transfer_vi_to_ir.zero_grad()
        optimizer_transfer_ir_to_vi.zero_grad()
        optimizer_decoder_1.zero_grad()
        optimizer_decoder_2.zero_grad()

        # 编码器提取特征
        vi_features = encoder_1(vi_images)   # [B, 512, H/8, W/8]
        ir_features = encoder_2(ir_images)   # [B, 512, H/8, W/8]

        # 跨域迁移: VI特征→IR域, IR特征→VI域
        vi_allin = transfer_vi_to_ir(vi_features, ir_features)  # [B, 512, H/8, W/8]
        ir_allin = transfer_ir_to_vi(ir_features, vi_features)  # [B, 512, H/8, W/8]

        # 重建: 用迁移后的特征重建对方模态的图像
        ir_out = decoder_2(vi_allin)  # 用VI→IR迁移特征重建IR图像
        vi_out = decoder_1(ir_allin)  # 用IR→VI迁移特征重建VI图像

        # 重建损失 = L1 + SSIM
        reir_loss = F.l1_loss(ir_out, ir_images) + ssim_loss(ir_out, ir_images)
        revi_loss = F.l1_loss(vi_out, vi_images) + ssim_loss(vi_out, vi_images)
        recon_loss = revi_loss + reir_loss

        # 域对抗损失: 让迁移后的特征欺骗判别器（目标为0.5，即域不变）
        domain_vi_a_pred = domain_discriminator(vi_allin)
        domain_ir_a_pred = domain_discriminator(ir_allin)

        domain_loss_fake = bce_loss(domain_vi_a_pred, domain_un_labels) + bce_loss(domain_ir_a_pred, domain_un_labels)
        domain_loss = domain_loss_fake

        # 均值-方差对齐损失: 约束两个迁移分支的特征分布一致
        mean_loss, variance_loss = loss_calculator(ir_allin, vi_allin)
        m_v_loss = mean_loss + variance_loss

        # 总损失 = 重建 + 域对抗 + 均值方差对齐
        total_loss = recon_loss + lambda_domain * domain_loss + lambda_mean_vari * m_v_loss
        total_loss.backward()

        optimizer_encoder_1.step()
        optimizer_encoder_2.step()
        optimizer_transfer_vi_to_ir.step()
        optimizer_transfer_ir_to_vi.step()
        optimizer_decoder_1.step()
        optimizer_decoder_2.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], total_loss: {total_loss.item():.4f}")


# ==================== 保存所有模型 ====================
save_model(encoder_1, encoder_1_path)
save_model(encoder_2, encoder_2_path)
save_model(transfer_vi_to_ir, transfer_vi_to_ir_path)
save_model(transfer_ir_to_vi, transfer_ir_to_vi_path)
save_model(domain_discriminator, domain_discriminator_path)
save_model(decoder_1, decoder_1_path)
save_model(decoder_2, decoder_2_path)
