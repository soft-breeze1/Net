import torch
from torch import nn

from Net.CFDN import CFDN
from utils.wavelet import DWT, IWT


class WaveletCFDN(nn.Module):
    def __init__(self):
        super(WaveletCFDN, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.cfdn = CFDN(num_channels=3)  # 修改为单网络处理三通道

    def forward(self, x):
        # 输入x形状: [B,3,H,W]
        y = self.dwt(x)
        B = x.size(0)
        y = y.view(B, 4, 3, y.size(-2), y.size(-1))

        # 分解子带
        ll = y[:, 0, :, :, :]  # [B,3,H//2,W//2]
        hl = y[:, 1, :, :, :]
        lh = y[:, 2, :, :, :]
        hh = y[:, 3, :, :, :]

        # 修改LL子带处理
        ll_opt, real_feat, imag_feat = self.cfdn(ll)  # 接收三个返回值

        # 重组子带
        y_opt = torch.stack([ll_opt, hl, lh, hh], dim=1)
        y_opt = y_opt.view(B * 4, 3, y_opt.size(-2), y_opt.size(-1))

        # 逆小波变换
        recon = self.iwt(y_opt)
        return recon, real_feat, imag_feat  # 返回三个值