import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM
from utils.wavelet import DWT, IWT,haar_dwt_ll


class CFDNLoss(nn.Module):
    def __init__(self, lambda_amp=1.0, lambda_phase=0.5, lambda_ssim=0.2, lambda_grad=0.1):
        super().__init__()
        self.lambda_amp = lambda_amp
        self.lambda_phase = lambda_phase
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.ssim = SSIM(win_size=11, channel=3, data_range=1.0)

        self.dwt = DWT()
        self.iwt = IWT()
        self.haar_dwt_ll = haar_dwt_ll

        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3))

    def forward(self, pred_img, target_img, fft_real_opt, fft_imag_opt):
        pred = (pred_img + 1) / 2
        target = (target_img + 1) / 2

        # --- 确保target_ll包含batch维度 [B, 3, H/2, W/2] ---
        target_ll= self.haar_dwt_ll(target_img)  # 必须包含batch维度
        B, C, H, W = target_ll.shape  # 验证维度：B=batch, C=3, H=H/2, W=W/2

        # 对目标LL子带进行傅里叶变换（保留batch维度）
        target_ll_fft = torch.fft.fft2(target_ll)
        real_target = target_ll_fft.real  # [B, 3, H, W]
        imag_target = target_ll_fft.imag  # [B, 3, H, W]

        # --- 振幅/相位计算（严格匹配维度）---
        amp_opt = torch.sqrt(fft_real_opt**2 + fft_imag_opt**2 + 1e-8)  # [B, 3, H, W]
        phase_opt = torch.atan2(fft_imag_opt, fft_real_opt + 1e-8)       # [B, 3, H, W]
        amp_target = torch.sqrt(real_target**2 + imag_target**2 + 1e-8)   # [B, 3, H, W]
        phase_target = torch.atan2(imag_target, real_target + 1e-8)       # [B, 3, H, W]

        loss_amp = F.l1_loss(amp_opt, amp_target)
        loss_phase = torch.mean(1 - torch.cos(phase_opt - phase_target))

        # --- 其他损失计算（不变）---
        loss_l1 = F.l1_loss(pred, target)
        loss_ssim = 1 - self.ssim(pred, target)
        loss_grad = self.compute_grad_loss(pred, target)

        total_loss = loss_l1 + self.lambda_amp*loss_amp + self.lambda_phase*loss_phase + \
                     self.lambda_ssim*loss_ssim + self.lambda_grad*loss_grad

        return {
            "total": total_loss,
            "loss_l1": loss_l1,
            "loss_amp": loss_amp,
            "loss_phase": loss_phase,
            "loss_ssim": loss_ssim,
            "loss_grad": loss_grad
        }

    def compute_grad_loss(self, pred, target):
        sobel_x_pred = self.apply_sobel(pred, self.sobel_x)
        sobel_y_pred = self.apply_sobel(pred, self.sobel_y)
        sobel_x_gt = self.apply_sobel(target, self.sobel_x)
        sobel_y_gt = self.apply_sobel(target, self.sobel_y)
        return (F.l1_loss(sobel_x_pred, sobel_x_gt) + F.l1_loss(sobel_y_pred, sobel_y_gt)) / 2

    def apply_sobel(self, x, sobel_kernel):
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        grad = F.conv2d(x_reshaped, sobel_kernel, padding=1)
        return grad.view(b, c, h, w)