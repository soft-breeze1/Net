import torch
import torch.nn as nn


class PSNRMetric(nn.Module):
    """
    峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）度量类，
    用于计算预测输出和真实标签之间的峰值信噪比，常用于评估图像质量。
    """

    def __init__(self):
        """
        初始化 PSNRMetric 类，调用父类 nn.Module 的初始化方法。
        """
        super(PSNRMetric, self).__init__()

    def forward(self, toutputs, tlabels):
        """
        前向传播函数，计算预测值与真实标签之间的峰值信噪比。

        参数:
            toutputs (torch.Tensor): 模型的预测输出张量，形状通常为 [batch_size, channels, height, width]。
            tlabels (torch.Tensor): 真实标签张量，形状应与 toutputs 相同。

        返回:
            float: 计算得到的峰值信噪比（PSNR）值。
        """
        data_range = 255  # 数据范围，对于图像数据，通常像素值范围是 0 到 255

        # 计算均方误差（MSE）：
        # 1. 计算预测值与真实标签的差的平方。
        # 2. 在通道、高度和宽度维度（dim=[1, 2, 3]）上取平均。
        # 3. 乘以数据范围的平方（data_range**2），将结果缩放到与原始数据范围相关的尺度。
        mse = torch.mean((toutputs - tlabels) ** 2, dim=[1, 2, 3]) * data_range ** 2

        # 计算峰值信噪比（PSNR）：
        # 根据 PSNR 的计算公式：PSNR = 10 * log10((data_range**2) / MSE)
        psnr = 10 * torch.log10(data_range ** 2 / mse)

        # 使用.item()方法将单元素张量转换为 Python 标量并返回
        return psnr.item()