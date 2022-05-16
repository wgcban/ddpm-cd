# Change detection head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from model.cd_modules.psp import _PSPModule
from model.cd_modules.se import ChannelSpatialSELayer

def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
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

class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim*len(time_steps), dim, 1)
            if len(time_steps)>1
            else None,
            nn.ReLU()
            if len(time_steps)>1
            else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class cd_head_v2(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None):
        super(cd_head_v2, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size       = img_size
        self.time_steps     = time_steps

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feats_A, feats_B):
        # Decoder
        lvl=0
        for layer in self.decoder:
            if isinstance(layer, Block):
                f_A = feats_A[0][self.feat_scales[lvl]]
                f_B = feats_B[0][self.feat_scales[lvl]]
                for i in range(1, len(self.time_steps)):
                    f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)
                    f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)
    
                diff = torch.abs( layer(f_A)  - layer(f_B) )
                if lvl!=0:
                    diff = diff + x
                lvl+=1
            else:
                diff = layer(diff)
                x = F.interpolate(diff, scale_factor=2, mode="bilinear")

        # Classifier
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))

        return cm

    