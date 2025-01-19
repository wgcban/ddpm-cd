import torch
import torch.nn as nn


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
