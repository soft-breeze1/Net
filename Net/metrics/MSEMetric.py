import torch
import torch.nn as nn


class MSEMetric(nn.Module):
    """
    计算均方误差(Mean Squared Error, MSE)的度量类
    常用于评估图像质量或模型预测结果与真实标签之间的差异
    """

    def __init__(self):
        """初始化MSEMetric类"""
        # 调用父类nn.Module的初始化方法
        super(MSEMetric, self).__init__()

    def forward(self, toutputs, tlabels):
        """
        前向传播函数，计算预测值与真实标签之间的MSE

        参数:
            toutputs (torch.Tensor): 模型的预测输出，形状通常为[batch_size, channels, height, width]
            tlabels (torch.Tensor): 真实标签，形状与toutputs相同

        返回:
            float: 计算得到的MSE值
        """
        # 数据范围，通常图像数据范围是[0, 255]，所以这里设为255
        # 这用于将像素值归一化到适当范围
        data_range = 255

        # 计算MSE：首先计算预测值与真实标签的差的平方
        # 然后在通道、高度和宽度维度上取平均(通过指定dim=[1, 2, 3])
        # 最后乘以data_range的平方，将结果缩放回原始数据范围
        mse = torch.mean((toutputs - tlabels) ** 2, dim=[1, 2, 3]) * data_range ** 2

        # 返回MSE值，使用.item()方法将单元素张量转换为Python标量
        return mse.item()