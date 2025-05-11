import random
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import argparse
from torch import nn
from Net.Net import WaveletCFDN
from Loss.CFDN_Loss import CFDNLoss
import torch.optim as optim
from utils.utils import findLastCheckpoint, batch_PSNR, batch_SSIM
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.data_RGB import get_training_data, get_validation_data

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    parser = argparse.ArgumentParser(description="WaveletCFDN Training")
    parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
    parser.add_argument("--patchSize", type=int, default=256, help="Patch size for training")
    parser.add_argument("--epochs", type=int, default=1600, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--save_weights", type=str, default=r"D:\EI\Net\models\wavelet_cfdn", help='Model save directory')
    parser.add_argument("--train_data", type=str, default=r'D:\EI\Net\data\UIEBD\train', help='Training data path')
    parser.add_argument("--val_data", type=str, default=r'D:\EI\Net\data\UIEBD\test', help='Validation data path')
    parser.add_argument("--use_GPU", type=bool, default=True, help='Use GPU flag')
    parser.add_argument("--decay", type=int, default=25, help='Learning rate decay interval')
    return parser.parse_args()


def main():
    opt = get_args()

    # 创建保存目录
    os.makedirs(opt.save_weights, exist_ok=True)

    # 固定随机种子
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # 数据加载
    print("Loading datasets...")
    train_dataset = get_training_data(opt.train_data, {'patch_size': opt.patchSize})
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                              num_workers=4, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(opt.val_data, {'patch_size': opt.patchSize})
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=True)

    # 模型初始化
    model = WaveletCFDN()
    print(f"Parameters total: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失函数
    criterion = CFDNLoss()

    # 设备配置
    device = torch.device("cuda" if opt.use_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 / 1024:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024 / 1024:.2f} MB")  # 关键修改点

    model = model.to(device)
    criterion = criterion.to(device)

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    milestones = [i for i in range(1, opt.epochs + 1) if i % opt.decay == 0]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # 加载检查点
    initial_epoch = findLastCheckpoint(save_dir=opt.save_weights) or 0
    if initial_epoch > 0:
        model_path = os.path.join(opt.save_weights, f'net_epoch{initial_epoch}.pth')
        model.load_state_dict(torch.load(model_path))
        print(f"从epoch {initial_epoch}恢复训练")

    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(initial_epoch, opt.epochs):
        print(f"\nEpoch {epoch + 1}/{opt.epochs}, LR: {scheduler.get_last_lr()[0]:.6f}")
        epoch_start = time.time()
        epoch_loss = 0.0

        # 训练阶段
        model.train()
        for i, (target, input, _) in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()

            # 数据转移到设备
            input = input.to(device)  # [B,3,H,W]
            target = target.to(device)  # [B,3,H,W]

            # 前向传播
            output, real_feat, imag_feat = model(input)

            # 添加形状打印
            # print(f"Model output shapes: real_feat{real_feat.shape}, imag_feat{imag_feat.shape}")
            # print(f"Target shapes: {target.shape}")

            # 计算损失
            loss_dict = criterion(output, target, real_feat, imag_feat)
            loss = loss_dict["total"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"\nBatch {i + 1}/{len(train_loader)}")
                print(f"Total loss: {loss.item():.4f}")
                print(f"L1: {loss_dict['loss_l1']:.4f} | Amp: {loss_dict['loss_amp']:.4f}")
                print(f"Phase: {loss_dict['loss_phase']:.4f} | SSIM: {loss_dict['loss_ssim']:.4f}")
                print(f"Grad: {loss_dict['loss_grad']:.4f}")

        scheduler.step()  # 每个epoch更新学习率

        # 验证阶段
        model.eval()
        psnr_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for target, input, _ in tqdm(val_loader, desc="Validating"):
                input = input.to(device)
                target = target.to(device)

                # 前向传播
                output, _, _ = model(input)

                # 转换到[0,1]范围
                output = (output + 1) / 2
                output = torch.clamp(output, 0.0, 1.0)
                target = (target + 1) / 2  # 修复：添加target范围转换

                # 计算指标
                psnr = batch_PSNR(output, target, 1.0)
                ssim = batch_SSIM(output, target, 1.0)
                psnr_total += psnr
                ssim_total += ssim

        # 计算平均指标（移到验证循环外）
        epoch_psnr = psnr_total / len(val_loader)
        epoch_ssim = ssim_total / len(val_loader)

        # 更新最佳指标并保存模型（修正缩进）
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            torch.save(model.state_dict(), os.path.join(opt.save_weights, 'net_best_psnr.pth'))
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            torch.save(model.state_dict(), os.path.join(opt.save_weights, 'net_best_ssim.pth'))

        # 保存模型
        save_path = os.path.join(opt.save_weights, f'net_epoch{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)
        torch.save(model.state_dict(), os.path.join(opt.save_weights, 'net_last.pth'))

        # 输出训练信息
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch Time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
        print(f"Train Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Validation PSNR: {epoch_psnr:.4f} | SSIM: {epoch_ssim:.4f}")
        print(f"Best PSNR: {best_psnr:.4f} | Best SSIM: {best_ssim:.4f}")

        # 记录训练日志
        log_msg = (f"Epoch {epoch + 1}/{opt.epochs}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                  f"Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"PSNR: {epoch_psnr:.4f}, "
                  f"SSIM: {epoch_ssim:.4f}\n")
        with open(os.path.join(opt.save_weights, 'train_log.txt'), 'a') as f:
            f.write(log_msg)

if __name__ == '__main__':
    main()