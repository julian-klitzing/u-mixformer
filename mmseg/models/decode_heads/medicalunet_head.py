import torch
import torch.nn as nn
import numpy as np
# from . import vit_seg_configs as configs
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ml_collections import ConfigDict

"""
This is the implementation of CNN-style UNet encoder used in the paper "TransUNet: Transformers Make Strong
Encoders for Medical Image Segmentation"
https://arxiv.org/pdf/2102.04306.pdf
"""


class Conv2dReLU(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        # x = self.up(x)
        # With interpolate forward pass also works in validation mode with varying input size
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


@MODELS.register_module()
class MedicalUNetHead(BaseDecodeHead): # nn.Module

    # def __init__(self, config):
    def __init__(self, **kwargs):
        # super().__init__()
        super().__init__(input_transform='multiple_select', **kwargs)

        self.config = ConfigDict(dict(
            hidden_size=256, # c4 dim of MiT
            n_skip=3,
            skip_channels=[160, 64, 32], # channels of MiT stages excluding c4 (256)
            decoder_channels=[64, 32, 16] # [128, 64, 32] 
        ))
        head_channels = 160 # channels of c3 of MiT (first one to concat)
        self.conv_more = Conv2dReLU(
            self.config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = self.config.decoder_channels
        skip_channels = self.config.skip_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(3-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[2-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        blocks.reverse()
        self.blocks = nn.ModuleList(blocks)
    
    # https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/add_models.html
    # says that this function must be implemented
    
    # def init_weights(self):
    #     pass

    def forward(self, hidden_states, features=None):

        x = hidden_states[-1]
        x = self.conv_more(x)
        for i in range(len(hidden_states) - 2, 0 - 1, -1):
            skip = hidden_states[i]
            decoder_block = self.blocks[i]
            x = decoder_block(x, skip=skip)
        
        x = self.cls_seg(x)
        return x


# CONFIGS = {
#     'ViT-B_16': configs.get_b16_config(),
#     'ViT-B_32': configs.get_b32_config(),
#     'ViT-L_16': configs.get_l16_config(),
#     'ViT-L_32': configs.get_l32_config(),
#     'ViT-H_14': configs.get_h14_config(),
#     'R50-ViT-B_16': configs.get_r50_b16_config(),
#     'R50-ViT-L_16': configs.get_r50_l16_config(),
#     'testing': configs.get_testing(),
# }
