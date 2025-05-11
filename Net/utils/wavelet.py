import torch
import torch.nn as nn
from PIL import Image


def Normalize(x):
    """
    将输入张量归一化到 [ymin, ymax] 范围。
    具体来说，该函数将输入张量的数值范围从其原始的最小值和最大值范围，
    映射到指定的 [ymin, ymax] 范围。

    参数:
    x (torch.Tensor): 输入的张量，其形状可以是任意的，但通常为 [batch_size, channels, height, width]。

    返回:
    torch.Tensor: 归一化后的张量，其形状与输入张量相同，且数值范围在 [ymin, ymax] 之间。
    """
    ymax = 255
    ymin = 0
    xmax = x.max()  # 获取输入张量 x 的最大值
    xmin = x.min()  # 获取输入张量 x 的最小值

    # 使用归一化公式：y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
    # 其中，x 是输入张量的每个元素，xmin 和 xmax 是输入张量的最小值和最大值，
    # ymin 和 ymax 是目标范围的最小值和最大值，y 是归一化后的结果。
    return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


def ensure_even_dimensions(image, mode='resize'):
    """
    确保图像的宽度和高度均为偶数。

    参数:
    image (PIL.Image): 输入的PIL图像
    mode (str): 处理模式，可选值:
                'resize' - 调整图像尺寸（默认）
                'crop' - 裁剪图像边缘
                'pad' - 填充图像边缘

    返回:
    PIL.Image: 尺寸调整后的图像
    """
    width, height = image.size
    new_width, new_height = width, height

    # 检查并调整宽度为偶数
    if width % 2 != 0:
        if mode == 'resize':
            new_width = width - 1
        elif mode == 'crop':
            new_width = width - 1
        elif mode == 'pad':
            new_width = width + 1

    # 检查并调整高度为偶数
    if height % 2 != 0:
        if mode == 'resize':
            new_height = height - 1
        elif mode == 'crop':
            new_height = height - 1
        elif mode == 'pad':
            new_height = height + 1

    # 根据不同模式处理图像
    if mode == 'resize':
        # 调整尺寸（可能会改变图像比例）
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif mode == 'crop':
        # 裁剪图像（保持中心区域）
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        return image.crop((left, top, right, bottom))
    elif mode == 'pad':
        # 填充图像（边缘添加像素）
        result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
        result.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
        return result

    return image  # 如果模式无效，返回原始图像


def dwt_init(x):
    """
    实现二维离散小波变换(Haar小波)
    将输入图像分解为四个子带:
    - LL: 低频近似系数
    - HL: 水平方向高频细节
    - LH: 垂直方向高频细节
    - HH: 对角线方向高频细节
    """
    # 隔行采样并缩放
    x01 = x[:, :, 0::2, :] / 2  # 偶数行
    x02 = x[:, :, 1::2, :] / 2  # 奇数行

    # 隔列采样
    x1 = x01[:, :, :, 0::2]  # 偶数行偶数列
    x2 = x02[:, :, :, 0::2]  # 奇数行偶数列
    x3 = x01[:, :, :, 1::2]  # 偶数行奇数列
    x4 = x02[:, :, :, 1::2]  # 奇数行奇数列

    # 计算四个子带
    # print("x1 size:", x1.size())
    # print("x2 size:", x2.size())
    # print("x3 size:", x3.size())
    # print("x4 size:", x4.size())
    x_LL = x1 + x2 + x3 + x4  # 低频近似
    x_HL = -x1 - x2 + x3 + x4  # 水平方向高频
    x_LH = -x1 + x2 - x3 + x4  # 垂直方向高频
    x_HH = x1 - x2 - x3 + x4  # 对角线方向高频

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)  # 在批次维度上拼接四个子带


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    """
    实现二维离散小波逆变换(Haar小波)
    将四个子带重新组合恢复原始图像
    """
    r = 2  # 重建因子
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width

    # 分割四个子带
    x1 = x[0:out_batch, :, :, :] / 2  # LL子带
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2  # HL子带
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2  # LH子带
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2  # HH子带

    # 初始化重建图像
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    # 从四个子带重建原始图像
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4  # 偶数行偶数列
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4  # 奇数行偶数列
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4  # 偶数行奇数列
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4  # 奇数行奇数列

    return h


class DWT(nn.Module):
    """二维离散小波变换模块"""

    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理操作，不需要梯度计算

    def forward(self, x):
        """执行小波变换"""
        return dwt_init(x)


class IWT(nn.Module):
    """二维离散小波逆变换模块"""

    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False  # 信号处理操作，不需要梯度计算

    def forward(self, x):
        """执行小波逆变换"""
        return iwt_init(x)

"""
对target图像做小波变换得到LL子带
"""
def haar_dwt_ll(x):
    """
    二维Haar小波变换，提取低频近似子带（LL）
    输入: x [B, C, H, W]（H/W必须为偶数）
    输出: ll [B, C, H/2, W/2]
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "输入尺寸H/W必须为偶数"

    # 行方向低通滤波（奇偶行平均）
    even_rows = x[:, :, 0::2, :]  # 偶数行 [B, C, H/2, W]
    odd_rows = x[:, :, 1::2, :]  # 奇数行 [B, C, H/2, W]
    row_avg = (even_rows + odd_rows) / 2  # 行方向低通输出 [B, C, H/2, W]

    # 列方向低通滤波（奇偶列平均）
    even_cols = row_avg[:, :, :, 0::2]  # 偶数列 [B, C, H/2, W/2]
    odd_cols = row_avg[:, :, :, 1::2]  # 奇数列 [B, C, H/2, W/2]
    ll = (even_cols + odd_cols) / 2  # 列方向低通输出（LL子带）

    return ll