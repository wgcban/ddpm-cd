# The decoder in our RSCD model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from skimage.segmentation import slic
from PIL import Image
import matplotlib.pyplot as plt

from cnn_layer import *
from slic2gcn import *


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    """
    Get the number of input layers to the change detection head.
    """
    in_channels = 0
    for scale in feat_scales:
        if scale < 3: # 256 x 256
            in_channels += inner_channel*channel_multiplier[0]
        elif scale < 6: # 128 x 128
            in_channels += inner_channel*channel_multiplier[1]
        elif scale < 9: # 64 x 64
            in_channels += inner_channel*channel_multiplier[2]
        elif scale < 12: # 32 x 32
            in_channels += inner_channel*channel_multiplier[3]
        elif scale < 15: # 16 x 16
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


class AlignSlTs(nn.Module):
    def __init__(self, inplane, kernel_size=3):
        super(AlignSlTs, self).__init__()
        self.down_h = EfficientDownsampling(inplane, inplane)
        self.down_l = EfficientDownsampling(inplane, inplane)
        self.flow_make = nn.Conv2d(inplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


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
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        self.time_steps = time_steps

        self.decoder = nn.ModuleList()

    def forward(self, feats_A, feats_B):
        tensor2slic = SuperpixelSegmentation()
        lvl = 0
        for layer in self.decoder:
            if isinstance(layer, Block):
                f_A = feats_A[0][self.feat_scales[lvl]]
                f_A = SuperpixelEncoder(f_A)
                f_B = feats_B[0][self.feat_scales[lvl]]
                f_B = SuperpixelEncoder(f_B)
                if len(self.time_steps) > 1:
                    for i in range(1, len(self.time_steps)):
                        f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)
                        f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)

                diff = torch.abs(layer(f_A) - layer(f_B))
                if lvl != 0:
                    diff = diff + x
                lvl += 1
            else:
                diff = layer(diff)
                x = F.interpolate(diff, scale_factor=2, mode="bilinear")
        feats_A = tensor2slic(feats_A)
        pass
