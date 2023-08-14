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

from typing import List, Tuple
from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from ..losses import accuracy
from numbers import Number

from IPython import embed


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
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
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

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
            x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
        else:
            x_ = y
            
        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4) #여기에다가 rollout을 넣는다면?
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CrossAttention2(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
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

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            # x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
            # x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
            x_ = self.norm(y)
            x_ = self.act(y)
        else:
            x_ = y
            
        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4) #여기에다가 rollout을 넣는다면?
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MultiCrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(sum(dim2) + dim1, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool1 = nn.AvgPool2d(pool_ratio[0], pool_ratio[0])
        self.pool2 = nn.AvgPool2d(pool_ratio[1], pool_ratio[1])
        self.pool3 = nn.AvgPool2d(pool_ratio[2], pool_ratio[2])
        self.sr1 = nn.Conv2d(dim2[0], dim2[0], kernel_size=1, stride=1)
        self.sr2 = nn.Conv2d(dim2[1], dim2[1], kernel_size=1, stride=1)
        self.sr3 = nn.Conv2d(dim2[2], dim2[2], kernel_size=1, stride=1)
        self.norm1 = nn.LayerNorm(dim2[0])
        self.norm2 = nn.LayerNorm(dim2[1])
        self.norm3 = nn.LayerNorm(dim2[2])
        self.norm4 = nn.LayerNorm(dim1)
        self.act = nn.GELU()
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

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2_1, N2_1, C2_1 = y[0].shape
        B2_2, N2_2, C2_2 = y[1].shape
        B2_3, N2_3, C2_3 = y[2].shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        x_1 = y[0].permute(0, 2, 1).reshape(B2_1, C2_1, H2[0], W2[0])
        x_1 = self.sr1(self.pool1(x_1)).reshape(B2_1, C2_1, -1).permute(0, 2, 1)
        x_1 = self.norm1(x_1)
        x_1 = self.act(x_1)
        x_2 = y[1].permute(0, 2, 1).reshape(B2_2, C2_2, H2[1], W2[1])
        x_2 = self.sr2(self.pool2(x_2)).reshape(B2_2, C2_2, -1).permute(0, 2, 1)
        x_2 = self.norm2(x_2)
        x_2 = self.act(x_2)
        x_3 = y[2].permute(0, 2, 1).reshape(B2_3, C2_3, H2[2], W2[2])
        x_3 = self.sr3(self.pool3(x_3)).reshape(B2_3, C2_3, -1).permute(0, 2, 1)
        x_3 = self.norm3(x_3)
        x_3 = self.act(x_3)
        x_4 = self.norm4(x)
        x_4 = self.act(x_4)

        x_ = torch.cat([x_1, x_2, x_3, x_4], dim=2)
        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4) #여기에다가 rollout을 넣는다면?
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MultiBlock(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        ##

        self.attn = MultiCrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        self.norm1 = norm_layer(dim1)
        self.norm2_1, self.norm2_2, self.norm2_3 = norm_layer(dim2[0]), norm_layer(dim2[1]), norm_layer(dim2[2])
        self.norm3 = norm_layer(dim1)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, y, H2, H1):
        x = x + self.drop_path(self.attn(self.norm1(x), [self.norm2_1(y[0]), self.norm2_2(y[1]), self.norm2_3(y[2])], H2, H2)) #self.norm2(y)이 F1에 대한 값
        x = x + self.drop_path(self.mlp(self.norm3(x), H1, H1))

        return x

class Block(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2)) #self.norm2(y)이 F1에 대한 값
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))

        # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
        # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x

class Block2(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention2(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2)) #self.norm2(y)이 F1에 대한 값
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))

        # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
        # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x

class MultiPlusCrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2+dim1, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
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

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
        x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        x_ = torch.cat([x_, x], dim=2)
        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4) #여기에다가 rollout을 넣는다면?
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class BlockPlus(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = MultiPlusCrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
            H2, W2 (int, int): Is height and width of input y.
            H1, W1 (int, int): Is height and width of input x.
        """
        # H2, W2 needed because y needs to be reshaped to apply pool
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
        # H1, W1 needed because x is still H1, W1 and needs to be reshaped to apply conv
        x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x

class CatKey(nn.Module):

    def __init__(self, pool_ratio=8, dim = 128, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.pool1 = nn.AvgPool2d(self.pool_ratio[1], self.pool_ratio[1], ceil_mode=True)
        self.sr1 = nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1)
        self.pool2 = nn.AvgPool2d(self.pool_ratio[2], self.pool_ratio[2], ceil_mode=True)
        self.sr2 = nn.Conv2d(dim[2], dim[2], kernel_size=1, stride=1)
        self.pool3 = nn.AvgPool2d(self.pool_ratio[3], self.pool_ratio[3], ceil_mode=True)
        self.sr3 = nn.Conv2d(dim[3], dim[3], kernel_size=1, stride=1)
        # self.norm = nn.LayerNorm(dim2)
        # self.act = nn.GELU()

    def forward(self, x):
        return torch.cat([x[0], self.sr1(self.pool1(x[1])), self.sr2(self.pool2(x[2])), self.sr3(self.pool3(x[3]))], dim=1)

@MODELS.register_module()
class FeedFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FeedFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 128

        self.attn_c4_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=8)
        self.attn_c3_c1 = Block(dim1=c3_in_channels, dim2=c1_in_channels, num_heads=5, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2_c1 = Block(dim1=c2_in_channels, dim2=c1_in_channels, num_heads=2, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=2)

        self.linear_fuse = ConvModule(
            in_channels=(c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

    
        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]

        _c4 = self.attn_c4_c1(c4, c1, h1, w1, h4, w4)
        # _c4 += c4
        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)

        _c3 = self.attn_c3_c1(c3, c1, h1, w1, h3, w3)
        # _c3 += c3
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)

        _c2 = self.attn_c2_c1(c2, c1, h1, w1, h2, w2)
        # _c2 += c2
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        _c2 = resize(_c2, size=(h1, w1), mode='bilinear', align_corners=False)

        _c1 = c1.permute(0, 2, 1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

@MODELS.register_module()
class FeedFormerHead_cat(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FeedFormerHead_cat, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 128

        self.attn_c4 = Block2(dim1=c4_in_channels, dim2=512, num_heads=8, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block2(dim1=c3_in_channels, dim2=512, num_heads=5, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block2(dim1=c2_in_channels, dim2=512, num_heads=2, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block2(dim1=c1_in_channels, dim2=512, num_heads=1, mlp_ratio=4,
                                drop_path=0.1, pool_ratio=1)
        self.cat_key = CatKey(pool_ratio=[1, 2, 4, 8], dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=(c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key([c4, c3, c2, c1])
        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]

        _c4 = self.attn_c4(c4, c_key, h4, w4, h4, w4)
        # _c4 += c4
        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)

        _c3 = self.attn_c3(c3, c_key, h4, w4, h3, w3)
        # _c3 += c3
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)

        _c2 = self.attn_c2(c2, c_key, h4, w4, h2, w2)
        # _c2 += c2
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        _c2 = resize(_c2, size=(h1, w1), mode='bilinear', align_corners=False)

        _c1 = self.attn_c1(c1, c_key, h4, w4, h1, w1)
        _c1 = c1.permute(0, 2, 1).reshape(n, -1, h1, w1)
        _c1 = resize(_c1, size=(h1, w1), mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x



class DeepVIB(nn.Module):
    def __init__(self, z_dim):
        """
        Deep VIB Model.

        Arguments:
        ----------
        input_shape : `int`
            Flattened size of image. (Default=784)
        output_shape : `int`
            Number of classes. (Default=10)
        z_dim : `int`
            The dimension of the latent variable z. (Default=256)
        """
        super(DeepVIB, self).__init__()
        self.z_dim  = z_dim

        # build encoder
        self.fc_mu  = nn.Linear(256, self.z_dim)
        self.fc_std = nn.Linear(256, self.z_dim)

    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        """
        Forward pass

        Parameters:
        -----------
        x : [batch_size,28,28]
        """
        # flattent image
        mu, std = self.fc_mu(x), F.softplus(self.fc_std(x), beta=1)
        encoder_out = self.reparameterise(mu, std) # sample latent based on encoder outputs
        return encoder_out, mu, std
    
@MODELS.register_module()
class FeedFormerHead32(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FeedFormerHead32, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 512

        self.attn_c4_c3 = Block(dim1=c4_in_channels, dim2=c3_in_channels, num_heads=8, mlp_ratio=4, #query:c4, key&value:c3
                                drop_path=0.1, pool_ratio=-2)
        self.attn_c3_c2 = Block(dim1=c4_in_channels, dim2=c2_in_channels, num_heads=8, mlp_ratio=4, #query:c3, key&value:c2
                                drop_path=0.1, pool_ratio=2)
        self.attn_c2_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4, #query:c2, key&value:c1
                                drop_path=0.1, pool_ratio=4)

        # self.se = SELayer(embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=(c4_in_channels * 4),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape
    
        # Upsampling to the next higher feature map to be fused (UNet style)
        c4 = resize(c4, size=(h3, w3), mode='bilinear', align_corners=False)
        n, _, h4, w4 = c4.shape

        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]
        _c4 = c4.flatten(2).transpose(1, 2) 

        _c3 = self.attn_c4_c3(_c4, c3, h3, w3, h4, w4)
        _c2 = self.attn_c3_c2(_c3, c2, h2, w2, h4, w4)
        _c1 = self.attn_c2_c1(_c2, c1, h1, w1, h4, w4)

        # _c4 = c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h4, w4)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h4, w4)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h4, w4)

        _c = self.linear_fuse(torch.cat([c4, _c3, _c2, _c1], dim=1))

        # _c = self.se(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

@MODELS.register_module()
class FeedFormerHead32_new(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FeedFormerHead32_new, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 512

        self.attn_c4_c3 = Block(dim1=c4_in_channels, dim2=c3_in_channels, num_heads=8, mlp_ratio=4, #query:c4, key&value:c3
                                drop_path=0.1, pool_ratio=2)
        self.attn_c3_c2 = Block(dim1=c4_in_channels, dim2=c2_in_channels, num_heads=8, mlp_ratio=4, #query:c3, key&value:c2
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4, #query:c2, key&value:c1
                                drop_path=0.1, pool_ratio=8)
        # self.attn_c4_c3 = Block(dim1=c4_in_channels, dim2=c3_in_channels, num_heads=8, mlp_ratio=4, #query:c4, key&value:c3
        #                         drop_path=0.1, pool_ratio=-2)
        # self.attn_c3_c2 = Block(dim1=c4_in_channels, dim2=c2_in_channels, num_heads=8, mlp_ratio=4, #query:c3, key&value:c2
        #                         drop_path=0.1, pool_ratio=2)
        # self.attn_c2_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4, #query:c2, key&value:c1
        #                         drop_path=0.1, pool_ratio=4)

        # self.se = SELayer(embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=(c4_in_channels * 4),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape
    
        # Upsampling to the next higher feature map to be fused (UNet style)
        c4 = resize(c4, size=(h3, w3), mode='bilinear', align_corners=False)
        n, _, h4, w4 = c4.shape

        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]
        _c4 = c4.flatten(2).transpose(1, 2) 

        _c3 = self.attn_c4_c3(_c4, c3, h3, w3, h4, w4)
        _c2 = self.attn_c3_c2(_c3, c2, h2, w2, h4, w4)
        _c1 = self.attn_c2_c1(_c2, c1, h1, w1, h4, w4)

        # _c4 = c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h4, w4)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h4, w4)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h4, w4)

        _c = self.linear_fuse(torch.cat([c4, _c3, _c2, _c1], dim=1))

        # _c = self.se(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


@MODELS.register_module()
class FeedFormerHeadUNet(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FeedFormerHeadUNet, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 512

        self.attn_c4_c3 = Block(dim1=c4_in_channels, dim2=c3_in_channels, num_heads=8, mlp_ratio=4, #query:c4, key&value:c3
                                drop_path=0.1, pool_ratio=2)
        self.attn_c3_c2 = Block(dim1=c4_in_channels, dim2=c2_in_channels, num_heads=8, mlp_ratio=4, #query:c3, key&value:c2
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4, #query:c2, key&value:c1
                                drop_path=0.1, pool_ratio=8)

        self.linear_fuse = ConvModule(
            in_channels=(c4_in_channels * 4),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]

        _c3 = self.attn_c4_c3(c4, c3, h3, w3, h4, w4)
        _c2 = self.attn_c3_c2(_c3, c2, h2, w2, h4, w4)
        _c1 = self.attn_c2_c1(_c2, c1, h1, w1, h4, w4)

        _c4 = c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h4, w4)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h4, w4)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h4, w4)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

@MODELS.register_module()
class FeedFormerHeadUNetPlus(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FeedFormerHeadUNetPlus, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 512

        self.attn_c4_c3 = BlockPlus(dim1=c4_in_channels, dim2=c3_in_channels, num_heads=8, mlp_ratio=4, #query:c4, key&value:c3
                                drop_path=0.1, pool_ratio=2)
        self.attn_c3_c2 = BlockPlus(dim1=c4_in_channels, dim2=c2_in_channels, num_heads=8, mlp_ratio=4, #query:c3, key&value:c2
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2_c1 = BlockPlus(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4, #query:c2, key&value:c1
                                drop_path=0.1, pool_ratio=8)

        self.linear_fuse = ConvModule(
            in_channels=(c4_in_channels * 4),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

    
        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]

        _c3 = self.attn_c4_c3(c4, c3, h3, w3, h4, w4)
        _c2 = self.attn_c3_c2(_c3, c2, h2, w2, h4, w4)
        _c1 = self.attn_c2_c1(_c2, c1, h1, w1, h4, w4)

        _c4 = c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h4, w4)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h4, w4)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h4, w4)

        # _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        # _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        # _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        # _c1 = resize(_c1, size=(h1,w1), mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


    
@MODELS.register_module()
class FeedFormerHead_new(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FeedFormerHead_new, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        # Hyperparameters
        beta   = 1e-3
        z_dim  = 256
        epochs = 200
        batch_size = 128
        learning_rate = 1e-4
        decay_rate = 0.97

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 512

        self.VIB = DeepVIB(z_dim = z_dim)

        self.attn_c4_c3 = Block(dim1=c4_in_channels, dim2=c3_in_channels, num_heads=8, mlp_ratio=4, #query:c4, key&value:c3
                                drop_path=0.1, pool_ratio=-2)
        self.attn_c3_c2 = Block(dim1=c4_in_channels, dim2=c2_in_channels, num_heads=8, mlp_ratio=4, #query:c3, key&value:c2
                                drop_path=0.1, pool_ratio=2)
        self.attn_c2_c1 = Block(dim1=c4_in_channels, dim2=c1_in_channels, num_heads=8, mlp_ratio=4, #query:c2, key&value:c1
                                drop_path=0.1, pool_ratio=4)

        self.linear_fuse = ConvModule(
            in_channels=(c4_in_channels * 4),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape
        
        # Upsampling to the next higher feature map to be fused (UNet style)
        c4 = resize(c4, size=(h3, w3), mode='bilinear', align_corners=False)
        n, _, h4, w4 = c4.shape

        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, patches]
        c4 = c4.flatten(2).transpose(1, 2) 

        c4_out, mu, std = self.VIB(c4)

        _c3 = self.attn_c4_c3(c4_out, c3, h3, w3, h4, w4)
        _c2 = self.attn_c3_c2(_c3, c2, h2, w2, h4, w4)
        _c1 = self.attn_c2_c1(_c2, c1, h1, w1, h4, w4)

        _c4 = c4_out.permute(0,2,1).reshape(n, -1, h4, w4)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h4, w4)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h4, w4)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h4, w4)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # _c = self.se(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x, mu, std
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, mu, std = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples, mu, std)
        return losses

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList, mu, std) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_kl':
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,
                        mu, std,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else: #cross entropy
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            else:
                if loss_decode.loss_name == 'loss_kl':
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        mu, std,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                    

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, _, _ = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)