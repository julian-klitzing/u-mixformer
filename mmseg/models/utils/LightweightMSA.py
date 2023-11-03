# Copyright (c) Nota AI GmbH All rights reserved.
from mmcv.cnn import (ConvModule, build_activation_layer)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp
from torch.cuda.amp import autocast

from .se_layer import SELayer
from typing import Union, Tuple

def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

class LightMLA(nn.Module):
    r"""Lightweight multi-scale linear attention (이게 핵심!!)"""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads = None,
        heads_ratio = 1.0,
        dim=8,
        use_bias=(False, False),
        inplace = False,
        conv_cfg=None,
        norm_cfg=(None, dict(type='BN')),
        act_cfg=(None, None),
        kernel_func=dict(type='ReLU'),
        scales=(5,),
        eps=1.0e-15,
    ):
        super(LightMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        self.dim = dim
        self.qkv = ConvModule(
            in_channels=in_channels,
            out_channels=3 * total_dim,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg[0],
            act_cfg=act_cfg[0],
            bias=use_bias[0]
            )

        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )

        # build activation layer
        kernel_func_ = kernel_func.copy()  # type: ignore
        # nn.Tanh has no 'inplace' argument
        if kernel_func_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
        ]:
            kernel_func_.setdefault('inplace', inplace)
        self.kernel_func = build_activation_layer(kernel_func_)

        self.proj = ConvModule(
            in_channels=total_dim * (1 + len(scales)),
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg[1],
            act_cfg=act_cfg[1],
            bias=use_bias[1]
            )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = qkv.size()

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = qkv.transpose(-1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = out.transpose(-1, -2)
        out = out.reshape((B, -1, H, W))
        return out

    def forward(self, x):
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        return self.proj(out)