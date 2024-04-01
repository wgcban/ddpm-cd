# The decoder in our RSCD model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from skimage.segmentation import slic
from PIL import Image
import matplotlib.pyplot as plt


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    """
    Get the number of input layers to the change detection head.
    """
    in_channels = 0
    for scale in feat_scales:
        if scale < 3: #256 x 256
            in_channels += inner_channel*channel_multiplier[0]
        elif scale < 6: #128 x 128
            in_channels += inner_channel*channel_multiplier[1]
        elif scale < 9: #64 x 64
            in_channels += inner_channel*channel_multiplier[2]
        elif scale < 12: #32 x 32
            in_channels += inner_channel*channel_multiplier[3]
        elif scale < 15: #16 x 16
            in_channels += inner_channel*channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class SuperpixelSegmentation(nn.Module):
    # TODO: add transform slic result into a tensor (B, C, H, W) -> (B, 1, sqrt(H), sqrt(W))
    def __init__(self, n_segments=100, compactness=10.0, sigma=1):
        super(SuperpixelSegmentation, self).__init__()
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
