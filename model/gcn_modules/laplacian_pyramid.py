import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPyramid(nn.Module):
    def __init__(self, num_levels, kernel_size=5, sigma=1.5):
        super(GaussianPyramid, self).__init__()
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.pyramid = nn.ModuleList([nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False) for _ in range(num_levels)])
        for layer in self.pyramid:
            nn.init.gaussian_(layer.weight, 0, sigma)

    def forward(self, x):
        pyramid = [F.conv2d(x, self.pyramid[0].weight, padding=self.kernel_size//2)]
        for i in range(1, self.num_levels):
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            pyramid.append(F.conv2d(x, self.pyramid[i].weight, padding=self.kernel_size//2))
        return pyramid


class LaplacianPyramid(nn.Module):
    def __init__(self, num_levels=5):
        super(LaplacianPyramid, self).__init__()
        self.num_levels = num_levels
        self.gaussian_pyramid = GaussianPyramid(num_levels)
        self._build_laplacian_pyramid(num_levels)

    def _build_laplacian_pyramid(self, num_levels):
        # 构建拉普拉斯金字塔的层
        laplacian_layers = []
        gaussian_pyramid = self.gaussian_pyramid.forward(x)  # 假设x是输入图像
        for i in range(1, num_levels):
            upsampled = F.interpolate(gaussian_pyramid[i], scale_factor=2, mode='bilinear', align_corners=True)
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_layers.append(laplacian)
        return laplacian_layers

    def forward(self, x):
        # 构建高斯金字塔
        gaussian_pyramid = self.gaussian_pyramid.forward(x)
        # 构建拉普拉斯金字塔
        laplacian_pyramid = self._build_laplacian_pyramid(self.num_levels)(x)
        # 抑制低频信息（例如，通过设置阈值）
        for i in range(self.num_levels):
            # 假设阈值为threshold，可以根据实际情况进行调整
            threshold = 0.1
            laplacian_pyramid[i] = torch.clamp(laplacian_pyramid[i], min=-threshold, max=threshold)
        # 重构图像（此处省略重构图像的代码）
        return laplacian_pyramid
