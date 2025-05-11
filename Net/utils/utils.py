from pytorch_msssim import ssim as ssim_pytorch
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
from collections import OrderedDict
import glob
import torch
import cv2
import re
import os
import torch
from torch import sqrt, erf, exp, sign
import math
from natsort import natsorted


"""
主要实现了图像处理和模型训练过程中常用的一些工具函数，包括图像量化、归一化、PSNR 和 SSIM 计算、目录创建、检查点加载等功能。
"""

# 将图像像素值量化到指定范围
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


# 将数据归一化到 [0, 1] 范围
def normalize(data):
    return data / 255.


# 创建多个目录，如果路径是列表则依次创建每个路径对应的目录
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


# 创建单个目录，如果目录不存在则创建
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 获取指定路径下与指定会话匹配的最后一个文件路径
def get_last_path(path, session):
    x = natsorted(glob(os.path.join(path, '*%s' % session)))[-1]
    return x


# 保存图像，将 RGB 图像转换为 BGR 格式后保存
def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# 计算两张图像的峰值信噪比（PSNR），此函数未完成实现
def calc_psnr(img1, img2):
    '''
        Calculate PSNR. img1 and img2 should be torch.Tensor and ranges within [0, 1].
    '''
    return


# 批量计算图像的 PSNR
def batch_PSNR(img, imclean, data_range):
    Img = img.data.detach().cpu().numpy().astype(np.float32)
    Iclean = imclean.data.detach().cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        """
        peak_signal_noise_ratio：这是 skimage.metrics 模块中的一个函数，用于计算两张图像之间的 PSNR 值。
        Iclean[i, :, :, :] 和 Img[i, :, :, :] 分别表示第 i 张真实图像和预测图像。
        """
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0]) # 将累加的 PSNR 值除以图像的数量（即批次大小），得到这批图像的平均 PSNR 值，并将其作为函数的返回值。


# 批量计算图像的结构相似性指数（SSIM）
def batch_SSIM(clean, noisy, data_range=1.0):
    """
    计算批量SSIM（使用pytorch_msssim，支持NCHW格式）
    输入：clean/noisy [B, C, H, W]，范围[0, 1]
    输出：平均SSIM值
    """
    ssim_total = 0.0
    win_size = 11  # 匹配pytorch_msssim默认窗口大小

    # 确保输入为NCHW格式（无需转换为HWC）
    for i in range(clean.size(0)):
        img_clean = clean[i].unsqueeze(0)  # 添加batch维度 [1, C, H, W]
        img_noisy = noisy[i].unsqueeze(0)

        # 调用pytorch_msssim的ssim（支持NCHW，自动处理多通道）
        ssim_val = ssim_pytorch(img_clean, img_noisy,
                                data_range=data_range,
                                win_size=win_size,
                                size_average=True)  # 显式指定参数

        ssim_total += ssim_val.item()  # 获取标量值

    return ssim_total / clean.size(0)


# 打印网络结构和参数数量
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


"""
主要功能是在指定的保存目录中查找最后保存的模型检查点文件，并返回该检查点对应的训练轮数（epoch）
"""
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'net_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*net_epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


# 计算两张图像的 PSNR
def cal_psnr(x_, x):
    mse = ((x_.astype(np.float) - x.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


# 将图像像素值归一化到指定的最小和最大范围内
def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)
    return img


# 根据指定范围对张量进行归一化
def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())


# 加载检查点到模型中
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


# 生成 Rizer 滤波器
def rizer_filters(size, scal, factor):
    # size: filter size
    # scal:Scale factor of multi-scale decomposition
    # factor: 一般取（1/2）^n (n = 1,2...)
    # f = exp(-w^2*q^2)/2
    # g = sqrt(1 - f**2)
    deter = factor * 2 ** (scal - 1)
    deter = torch.tensor(deter).cuda()
    w = []
    for i in range(-math.floor(size / 2), math.ceil(size / 2)):
        w.append(i)
    w = torch.tensor(w).cuda()
    # w的取值范围是（-pi,pi）
    pi = 3.1415926
    w = w * 2 * pi / size
    w = torch.fft.ifftshift(w)
    length = len(w)
    w = w.reshape(length, 1)

    f = exp(-(w ** 2) * (deter ** 2) / 2)
    g = sqrt(1 - f ** 2)
    gh = sign(w) * g * (-1j)

    f = f[:, 0]
    g = g[:, 0]
    gh = gh[:, 0]
    return f, g, gh