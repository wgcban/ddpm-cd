# The decoder in our RSCD model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from skimage.segmentation import slic
from PIL import Image
import matplotlib.pyplot as plt


class SuperpixelSegmentation:
    def __init__(self, n_segments=100, compactness=10.0, sigma=1):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma

    def segment(self, image_tensor):
        image_np = image_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        labels = slic(image_np, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma)
        labels_tensor = torch.from_numpy(labels).long()
        return labels_tensor

class cd_gcn_head(nn.Module):
    """
    The decoder in our GCN-RSCD model
    """
    def __init__(self,
                 feat_scales,
                 out_channels=2,
                 inner_channel=None,
                 channel_multiplier=None,
                 img_size=256,
                 time_steps=None):
        super(cd_gcn_head, self).__init__()

    def forward(self, feats_A, feats_B):
        pass

# 初始化超像素分割器
segmenter = SuperpixelSegmentation(n_segments=100, compactness=10.0, sigma=1)

# 生成随机的 Tensor 作为输入
input_tensor = torch.randn(1, 3, 256, 256)  # 生成一个大小为 256x256 的 RGB 图像

# 对图像进行超像素分割
segmented_image = segmenter.segment(input_tensor)
print(segmented_image.size())

# 显示分割结果
plt.imshow(segmented_image.squeeze(), cmap='viridis')
plt.show()
