import torch
import torch.nn as nn

from slic2gcn import *


class EfficientDownsampling(nn.Module):
    def __init__(self, inplane, outplane, stride=1, padding=0, norm_layer=None):
        super(EfficientDownsampling, self).__init__()
        self.depthwise_conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=1, padding=padding)
        self.pointwise_conv1 = nn.Conv2d(inplane, outplane, kernel_size=1, stride=1, padding=0)
        self.norm1 = norm_layer(outplane) if norm_layer else nn.Identity()

        self.depthwise_conv2 = nn.Conv2d(outplane, outplane, kernel_size=3, groups=outplane, stride=stride, padding=padding)
        self.pointwise_conv2 = nn.Conv2d(outplane, outplane, kernel_size=1, stride=1, padding=0)
        self.norm2 = norm_layer(outplane) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.norm1(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.norm2(x)
        x = nn.ReLU(inplace=True)(x)

        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
