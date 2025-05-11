import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    """
    生成一维高斯窗函数
    Args:
        window_size: 窗口大小
        sigma: 高斯分布的标准差
    Returns:
        归一化的一维高斯窗函数
    """
    # 计算每个位置的高斯值
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # 归一化处理
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    创建二维高斯窗用于SSIM计算
    Args:
        window_size: 窗口大小
        channel: 输入图像的通道数
    Returns:
        扩展到指定通道数的二维高斯窗
    """
    # 生成一维高斯窗并扩展为二维
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 扩展到指定通道数并设置为可训练变量
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    计算两张图像的SSIM（结构相似性）值
    Args:
        img1: 第一张图像
        img2: 第二张图像
        window: 高斯窗
        window_size: 窗口大小
        channel: 通道数
        size_average: 是否对结果取平均
    Returns:
        SSIM值或SSIM图
    """
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # 计算均值的平方和乘积
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 常数项，防止分母为零
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 计算SSIM图
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 根据size_average参数决定返回均值还是逐元素值
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMMetric(torch.nn.Module):
    """
    SSIM（结构相似性）度量模块，用于评估两张图像的相似性
    """

    def __init__(self, window_size=11, size_average=True):
        """
        初始化SSIM度量模块
        Args:
            window_size: 高斯窗口大小，默认为11
            size_average: 是否对结果取平均，默认为True
        """
        super(SSIMMetric, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        # 创建初始高斯窗
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        计算两张图像的SSIM值
        Args:
            img1: 第一张图像
            img2: 第二张图像
        Returns:
            SSIM值
        """
        # 获取图像通道数
        (_, channel, _, _) = img1.size()

        # 检查是否需要更新窗口以匹配输入图像的通道数和设备
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            # 如果在GPU上运行，将窗口移到相应设备
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            # 更新窗口和通道数
            self.window = window
            self.channel = channel

        # 计算SSIM值并返回标量
        val = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return val.item()