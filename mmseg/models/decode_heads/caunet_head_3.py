# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from ..utils import resize
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
import math
from timm.models.layers import DropPath, trunc_normal_

from ..utils import nchw_to_nlc, nlc_to_nchw
import torch.nn.functional as F

from IPython import embed

# new imports
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer

"""
CrossAttentionHead3 does upsampling on H, W at each stage but downamples channels (by dimension of linear projection for queries).
Also the bigger features (key, value) are reduced by pooling ratio, as applied in SegFormer EfficientSelfAttention.

--> Forward/Backward size for (1, 3, 512, 512) is  
"""


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, norm_cfg=dict(type='LN')):
        """
        Args:
            dim1 (int): Number channels C of the feature map with more channels.
            dim2 (int): Number channels C of the feature map with fewer channels.
        """    
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # ----- Change now that Q gets compressed to smaller size
        # self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        # self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)

        self.q = nn.Linear(dim1, dim2, bias=qkv_bias) # Changed from dim1, dim1 to dim1, dim2
        self.kv = nn.Linear(dim2, dim2 * 2, bias=qkv_bias) # Changed from dim2, dim1 * 2 to dim2, dim2 * 2
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim2, dim2) # Changed from dim1, dim1 to dim2, dim2
        self.proj_drop = nn.Dropout(proj_drop)

        # self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        # self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)


        # ----- Introduce the sequence reduction from SegFromer to reduce the K,V for improving efficiacy ------- 
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=dim2,
                out_channels=dim2,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, dim2)[1]
            self.act = nn.GELU() #(SegFormer doesn't use it too)
 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2, H1, W1):
        """
        Args:
            x (nn.Tensor): Is the data of shape (B, L, D) to be projected to Q.
            y (nn.Tensor): Is the data of shape (B, L, D) to be projected to K, V.
            H2, W2 (int, int): Is height and width of input x. (size for query part)
            H1, W1 (int, int): Is height and width of input y. (size for key part)
        """
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3) # Changed C1 to C2

        # Skip the pooling (i.e. downsampling) as we are upsampling the smaller featuremap instead
        # x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
        # x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
        # x_ = self.norm(x_) --> instead use y directly in the next line

        # -------- Taken from SegFormer & modified ---------
        if self.sr_ratio > 1:
            x_ = nlc_to_nchw(y, (H1, W1))
            x_ = self.sr(x_)
            x_ = nchw_to_nlc(x_)
            x_ = self.norm(x_)
            x_ = self.act(x_)
        else:
            x_ = y
        # ---------------------------------------

        # Do we even still need the norm when pooling is skipped?
        # x_ = self.norm(y)
        # x_ = self.act(x_)

        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) # Changed C1 to C2
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C2) # Changed C1 to C2

        # Somewhere here an error occurs!
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1) # Changed from dim1 to dim2 (But I think atually it worked work previous stage with old setting only for stage 2 it dindn't work anymore) --> back to dim1
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim2) # Changed from dim1 to dim2

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, sr_ratio=sr_ratio)
        self.proj_res = nn.Linear(dim1, dim2) # Introduced new projection for residual connection (otherwise non-matching shapes) --> Only now this in no real residual connection anymore ...
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim2, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) # Changed from dim1 to dim2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2, H1, W1):
        x = self.proj_res(x) + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2, H1, W1)) 
        x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1)) 

        return x

@MODELS.register_module()
class CrossAttentionUNetHead3(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(CrossAttentionUNetHead3, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 32 # Changed to 32 because this is our the embed dim at the final decoder stage (and actually in all others as well)

        # general recipe:
        # self.attn_cSmaller_cGreater = Block (dim1=cSmaller, dim2=cGreater, num_heads=Copy, mlp_ratio=Copy, drop_path=Copy, pool_ratio=Adjust) 

        # self.attn_c4_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4,
        #                         drop_path=0.1, pool_ratio=8)
        # self.attn_c3_c1 = Block(dim1=c3_in_channels, dim2=c1_in_channels, num_heads=5, mlp_ratio=4,
        #                         drop_path=0.1, pool_ratio=4)
        # self.attn_c2_c1 = Block(dim1=c2_in_channels, dim2=c1_in_channels, num_heads=2, mlp_ratio=4,
        #                         drop_path=0.1, pool_ratio=2)


        # Current problem num_channels are staying the same (so very high --> high compute; pool operation is currently deactivated!
        self.attn_c4_c3 = Block(dim1=c4_in_channels, dim2=c3_in_channels, num_heads=8, mlp_ratio=4,
                                drop_path=0.1, sr_ratio=2)
        self.attn_d1_c2 = Block(dim1=c3_in_channels, dim2=c2_in_channels, num_heads=4, mlp_ratio=4, # cannot be 4 here because needs to be devideable by emded_dim (256) # changed dim1 to c3_in_channels
                                drop_path=0.1, sr_ratio=4)
        self.attn_d2_c1 = Block(dim1=c2_in_channels, dim2=c1_in_channels, num_heads=2, mlp_ratio=4, # changed dim1 to c2_in_channels
                                drop_path=0.1, sr_ratio=8)

        # Not needed as no fusion is used
        # self.linear_fuse = ConvModule(
        #     in_channels=(c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels),
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='SyncBN', requires_grad=True)
        # )

        # Exchanged by inbuild self.cls_seg function
        # self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape
        
        # -------- First decoder stage ---------
        # Upsampling to the next higher feature map to be fused (UNet style)
        c4 = resize(c4, size=(h3, w3), mode='bilinear', align_corners=False)
        # n, _, h4, w4 = c4.shape # perhaps update values after resize? --> if is on x will be [4096, 20] not [1024, 160] 
        c4 = nchw_to_nlc(c4)
        c3 = nchw_to_nlc(c3)
        x = self.attn_c4_c3(c4, c3, h4, w4, h3, w3)
        h_d1, w_d1 = h3, w3 # inital 
        x = nlc_to_nchw(x, (h_d1, w_d1))

        # -------- Second decoder stage ---------
        x = resize(x, size=(h2, w2), mode='bilinear', align_corners=False)
        x = nchw_to_nlc(x)
        c2 = nchw_to_nlc(c2)
        x = self.attn_d1_c2(x, c2, h_d1, w_d1, h2, w2)
        h_d2, w_d2 = h2, w2 # inital 
        x = nlc_to_nchw(x, (h_d2, w_d2))

        # -------- Third decoder stage ---------
        x = resize(x, size=(h1, w1), mode='bilinear', align_corners=False)
        x = nchw_to_nlc(x)
        c1 = nchw_to_nlc(c1)
        x = self.attn_d2_c1(x, c1, h_d2, w_d2, h1, w1)
        h_d3, w_d3 = h1, w1 # inital 
        x = nlc_to_nchw(x, (h_d3, w_d3))


        # x = self.dropout(x)
        # x = self.linear_pred(x) use conv_seg provided of the parent class instead (otherwise cuda not all params are used issue)
        x = self.cls_seg(x)

        return x