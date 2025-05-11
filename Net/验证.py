import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# 导入自定义的SSIMMetric类
from metrics.SSIMMetric import SSIMMetric

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei"]  # 只使用SimHei字体，避免找不到字体的错误
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 确保可以导入wavelet.py中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wavelet import DWT, IWT, ensure_even_dimensions, Normalize


def main(image_path):
    """
    主函数：读取图片，执行小波变换和逆变换，并显示结果
    """
    try:
        # 读取并预处理图像
        image = Image.open(image_path).convert('RGB')
        print(f"原始图像尺寸: {image.size}")  # 打印原始图像尺寸
        image = ensure_even_dimensions(image, mode='resize')

        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备（CPU或GPU）
        img_tensor = img_tensor.to(device)  # 将张量移动到设备上

        # 创建小波变换和逆变换模块
        dwt = DWT().to(device)  # 将模块移动到设备上
        iwt = IWT().to(device)  # 将模块移动到设备上

        # 执行小波变换
        dwt_result = dwt(img_tensor)

        # 分离四个子带
        batch_size, channels, height, width = img_tensor.size()
        LL = dwt_result[0:1, :, :, :]  # 低频近似
        HL = dwt_result[1:2, :, :, :]  # 水平方向高频
        LH = dwt_result[2:3, :, :, :]  # 垂直方向高频
        HH = dwt_result[3:4, :, :, :]  # 对角线方向高频

        # 创建SSIMMetric实例
        ssim_metric = SSIMMetric()

        # 确保LL子带和原始图像尺寸一致（这里假设需要上采样LL子带）
        LL = torch.nn.functional.interpolate(LL, size=(height, width), mode='bicubic', align_corners=False)

        # 计算SSIM值
        ssim_value = ssim_metric(img_tensor, LL)

        # 执行逆变换
        reconstructed = iwt(dwt_result)

        #将张量移回CPU并转换为numpy数组进行可视化
        LL = Normalize(LL[0]).permute(1, 2, 0).byte().cpu().numpy()
        HL = Normalize(HL[0]).permute(1, 2, 0).byte().cpu().numpy()
        LH = Normalize(LH[0]).permute(1, 2, 0).byte().cpu().numpy()
        HH = Normalize(HH[0]).permute(1, 2, 0).byte().cpu().numpy()
        reconstructed = reconstructed[0].permute(1, 2, 0).byte().cpu().numpy()

        # 准备显示图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 显示原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        # 显示四个子带
        axes[0, 1].imshow(LL)
        axes[0, 1].set_title(f'LL (低频近似)\nSSIM: {ssim_value:.4f}')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(HL)
        axes[0, 2].set_title('HL (水平高频)')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(LH)
        axes[1, 0].set_title('LH (垂直高频)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(HH)
        axes[1, 1].set_title('HH (对角线高频)')
        axes[1, 1].axis('off')

        # 显示重建图像
        axes[1, 2].imshow(reconstructed)
        axes[1, 2].set_title('重建图像')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        print("小波变换和逆变换处理完成！")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    # 指定要处理的图片路径，这里使用相对路径，也可以改为绝对路径
    image_path = r"D:\EI\Net\data\UIEBD\test\input\797_img_.png"  # 请替换为实际的图片路径

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到图片文件 '{image_path}'")
        print("请确保指定的图片路径正确。")
    else:
        main(image_path)