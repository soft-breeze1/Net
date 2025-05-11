import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift


# ================== 多通道傅里叶处理器 ==================
class FourierProcessor(nn.Module):
    def forward(self, x):
        # 输入x形状: [B, C, H, W] (C=3表示RGB三通道)
        channels = x.size(1)
        freq_features = []

        # 对每个通道分别处理
        for c in range(channels):
            single_channel = x[:, c:c + 1, :, :]  # [B,1,H,W]
            f = fft2(single_channel)
            f_shifted = fftshift(f, dim=(-2, -1))
            real = f_shifted.real
            imag = f_shifted.imag
            freq_features.extend([real, imag])

        return torch.cat(freq_features, dim=1)  # [B, C*2, H, W] （3通道输入生成6通道频域特征）


# ================== 多通道残差块 ==================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=channels),  # 分组归一化
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=channels)
        )

    def forward(self, x):
        return x + self.conv(x)


# ================== 跨通道注意力机制 ==================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_wise(x)


# ================== 多通道分支网络 ==================
class AmplitudeBranch(nn.Module):
    def __init__(self, in_channels=6):  # 3通道输入对应6个频域通道
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.resblocks = nn.Sequential(
            *[ResBlock(64) for _ in range(4)]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.resblocks(x)
        return self.out_conv(x)


class PhaseBranch(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.att = ChannelAttention(64)
        self.out_conv = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.att(x)
        return self.out_conv(x)


# ================== 跨注意力模块优化 ==================
class CrossAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q_conv = nn.Conv2d(dim, dim, 1)
        self.kv_conv = nn.Conv2d(dim, dim * 2, 1)
        self.scale = dim ** -0.5

    def forward(self, amp_feat, phase_feat):
        B, C, H, W = amp_feat.shape

        # Query生成
        q = self.q_conv(amp_feat).view(B, C, -1).permute(0, 2, 1)  # [B,HW,C]

        # Key/Value生成
        kv = self.kv_conv(phase_feat)
        k, v = kv.chunk(2, dim=1)  # 各为[B,C,H,W]
        k = k.view(B, C, -1)  # [B,C,HW]
        v = v.view(B, C, -1).permute(0, 2, 1)  # [B,HW,C]

        # 注意力计算
        att = torch.bmm(q, k) * self.scale  # [B,HW,HW]
        att = F.softmax(att, dim=-1)

        out = torch.bmm(att, v).permute(0, 2, 1).view(B, C, H, W)
        return amp_feat + out  # 残差连接


# ================== 多通道CFDN网络 ==================
class CFDN(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.fourier = FourierProcessor()
        self.amp_branch = AmplitudeBranch(in_channels=num_channels * 2)  # 6通道输入（3实+3虚）
        self.phase_branch = PhaseBranch(in_channels=num_channels * 2)
        self.cross_att = CrossAttention()
        self.final_fusion = nn.Sequential(
            nn.Conv2d(64, num_channels, 3, padding=1),  # 输出3通道图像
            nn.Tanh()
        )

        # 关键修改：频域特征提取层输出3通道（对应3个原始通道的实部/虚部）
        self.real_extract = nn.Conv2d(64, num_channels, 1)  # 输出3通道实部特征
        self.imag_extract = nn.Conv2d(64, num_channels, 1)  # 输出3通道虚部特征

    def forward(self, x):
        freq_features = self.fourier(x)  # [B, 6, H, W]（3实部+3虚部）

        amp_feat = self.amp_branch(freq_features)  # [B, 64, H, W]
        phase_feat = self.phase_branch(freq_features)  # [B, 64, H, W]

        fused = self.cross_att(amp_feat, phase_feat)
        recon = self.final_fusion(fused)  # [B, 3, H, W]（重建图像）

        # 显式提取频域特征（3通道实部/虚部，与原始通道数一致）
        real_feat = self.real_extract(amp_feat)  # [B, 3, H, W]（3个通道的实部）
        imag_feat = self.imag_extract(phase_feat)  # [B, 3, H, W]（3个通道的虚部）

        return recon, real_feat, imag_feat

