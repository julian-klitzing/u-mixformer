import torch
import torch.nn as nn
import numpy as np
# from . import vit_seg_configs as configs

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ml_collections import ConfigDict


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
        x = self.up(x)
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
class TransUNetHead(BaseDecodeHead): # nn.Module

    # def __init__(self, config):
    def __init__(self, **kwargs):
        # super().__init__()
        super().__init__(input_transform='multiple_select', **kwargs)
        self.config = ConfigDict(dict(
            hidden_size=768,
            n_skip=3,
            skip_channels=[512, 256, 64, 16],
            decoder_channels=[256, 128, 64, 16]
        ))
        head_channels = 512
        self.conv_more = Conv2dReLU(
            self.config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = self.config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
    
    # https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/add_models.html
    # says that this function must be implemented
    
    # def init_weights(self):
    #     pass

    def forward(self, hidden_states, features=None):
        # Here should now arrive the 4 stages of the MixedVision Transformer
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
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
