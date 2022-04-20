# HSI encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class hsi_e(nn.Module):
    def __init__(self, in_channels=64, mid_channels=16, out_channels=64, bias=True):
        super(hsi_e, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=bias)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(2 * mid_channels, mid_channels, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(3 * mid_channels, mid_channels, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(4 * mid_channels, out_channels, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))

        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x1, x2, x3, x4), 1))
        return x5