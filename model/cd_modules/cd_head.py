# Change detection head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

#get input feature maps
def get_in_channels(feat_scales):
    if feat_scales <= 3:
        in_channels = (feat_scales)*64
    elif feat_scales <=7:
        in_channels = 4*64 + (feat_scales-4)*128
    elif feat_scales <=9:
        in_channels = 4*64 + 3*128 + (feat_scales-7)*256
    elif feat_scales <=11:
        in_channels = 4*64 + 2*128 + 3*256 + 3*512
    elif feat_scales <=14:
        in_channels = 4*64 + 2*128 + 3*256 + 3*512 + 3*512
    else:
        print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    
    return in_channels
    

class cd_head(nn.Module):
    """MLP change detection head."""

    def __init__(self, feat_scales, out_channels=2):
        super(cd_head, self).__init__()

        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales)

        # Convolutional layer to reduce the feature dimention
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)

        # Convolutional layer to get final prediction map
        self.conv_cd_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_cd_2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

        self.relu = torch.nn.ReLU()

    def forward(self, feats_A, feats_B): 
        
        input_A = feats_A[0]
        input_B = feats_B[0]

        for i in range(1, self.feat_scales):
            if feats_A[i].size(2) != feats_A[0].size(2):
                f_A = F.interpolate(feats_A[i], size=(feats_A[0].size(2), feats_A[0].size(3)), mode="bilinear")
                f_B = F.interpolate(feats_B[i], size=(feats_B[0].size(2), feats_B[0].size(3)), mode="bilinear")
            else:
                f_A = feats_A[i]
                f_B = feats_B[i]
            
            input_A = torch.cat((input_A, f_A), dim=1)
            input_B = torch.cat((input_B, f_B), dim=1)

        cm = torch.abs(self.conv1(input_A)-self.conv1(input_B))

        cm = self.relu(self.conv_cd_1(cm))
        cm = self.conv_cd_2(cm)

        return cm

    