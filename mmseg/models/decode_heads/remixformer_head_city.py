# ---------------------------------------------------------------
# Copyright (c) 2021, Nota AI GmbH. All rights reserved.
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from ..utils import resize
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
import math
from timm.models.layers import DropPath, trunc_normal_

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

class CatKey(nn.Module):
    def __init__(self, pool_ratio=[1,2,4,8], dim=[256,160,64,32]):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.ModuleList([nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])
        self.pool_list = nn.ModuleList([nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])

    def forward(self, x):
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                out_list.append(x[i])
        return torch.cat(out_list, dim=1)

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
            self.pool1 = nn.AvgPool2d(2, 2) #query
            self.pool2 = nn.AvgPool2d(pool_ratio * 2, pool_ratio * 2) #key&value
            self.sr1 = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
            self.sr2 = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)
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

    def forward(self, x, y, H2, W2, H1, W1):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B1, C1, H1, W1)
        x_ = self.sr1(self.pool1(x_)).reshape(B1, C1, -1).permute(0, 2, 1)
        x_ = self.norm1(x_)
        x_ = self.act(x_)
        N1 = N1 // (2 * 2)
        q = self.q(x_).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        # y_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
        # y_ = self.sr2(self.pool2(y_)).reshape(B2, C2, -1).permute(0, 2, 1)
        # y_ = self.norm2(y_)
        y_ = self.norm2(y)
        y_ = self.act(y_)
        kv = self.kv(y_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).view(B1, C1, H1 // 2, W1 // 2)
        x = resize(x, size=(H1, W1), mode='bilinear', align_corners=False)
        x = x.flatten(2).transpose(1, 2)

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
        x = x + self.drop_path(self.attn(x, y, H2, W2, H1, W1)) #self.norm2(y)이 F1에 대한 값
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))

        # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
        # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x

@MODELS.register_module()
class APFormerHeadCity(BaseDecodeHead):
    """
    Attention-Pooling Former
    """
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHeadCity, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4,
                                drop_path=0.1, pool_ratio=1)

        pool_ratio = [i * 2 for i in pool_ratio]
        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
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

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c4 = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c3 = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c2 = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c1 = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1, c_key, h4, w4, h1, w1)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x