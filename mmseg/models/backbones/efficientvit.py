# Copyright (c) Nota AI GmbH. All rights reserved.
import warnings

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import Conv2dAdaptivePadding
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import is_tuple_of
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.registry import MODELS
from ..utils import InvertedResidualV3 as InvertedResidual
from ..utils import InvertedResidualV3_bn, LightMLA

class EfficientViTBlock(BaseModule):
    """Inverted Residual Block for MobileNetV3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution.
            Default: 3.
        stride (int): The stride of the depthwise convolution. Default: 1.
        se_cfg (dict): Config dict for se layer. Default: None, which means no
            se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels. Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        heads_ratio = 1.0,
        dim=32,
        expand_ratio = 4,
        se_cfg=None,
        with_expand_conv=True,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        with_cp=False):

        super(EfficientViTBlock, self).__init__()

        self.context_module = LightMLA(
            in_channels=in_channels,
            out_channels=out_channels,
            heads_ratio=heads_ratio,
            dim=dim
            )

        mid_channels = in_channels*expand_ratio
        self.local_module = InvertedResidualV3_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            se_cfg=se_cfg,
            with_expand_conv=(in_channels != mid_channels),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp)

    def forward(self, x):
        x = self.context_module(x) + x
        x = self.local_module(x)
        return x


@MODELS.register_module()
class EfficientViT(BaseModule):
    """EfficientViT backbone.

    This backbone is the improved implementation of `Searching for EfficientViT

    Args:
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        out_indices (tuple[int]): Output from which layer.
            Default: (0, 1, 12).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 width_list = [16, 32, 64, 128, 256], #example: B1
                 depth_list = [1, 2, 3, 3, 4],
                 dim = 16,
                 expand_ratio = 4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 reduction_factor=1,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        assert isinstance(reduction_factor, int) and reduction_factor > 0
        assert is_tuple_of(out_indices, int)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.reduction_factor = reduction_factor
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.width_list = width_list
        self.depth_list = depth_list
        self.expand_ratio = expand_ratio
        self.dim = dim
        self.kernel_size = 3

        self.layers = self._make_layer()

    def _make_layer(self):
        layers = ModuleList()

        # build stem layer
        stem0 = ConvModule(
            in_channels=3,
            out_channels=self.width_list[0],
            kernel_size=self.kernel_size,
            stride=2,
            padding=1,
            conv_cfg=dict(type='Conv2dAdaptivePadding'),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        stem_block = ModuleList()
        for i in range(self.depth_list[0]):
            layer = InvertedResidual(
                in_channels=self.width_list[0],
                out_channels=self.width_list[0],
                mid_channels=self.width_list[0],
                kernel_size=self.kernel_size,
                stride=1,
                se_cfg=None,
                with_expand_conv=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp)
            stem_block.append(layer)
        in_channels = self.width_list[0]
        layers.append(ModuleList([stem0, stem_block]))

        # build the first two stage (1 and 2) (like MobileNet without efficientViT module)
        for w, d in zip(self.width_list[1:3], self.depth_list[1:3]):
            block = ModuleList()
            for i in range(d):
                # if with_se:
                #     se_cfg = dict(
                #         channels=mid_channels,
                #         ratio=4,
                #         act_cfg=(dict(type='ReLU'),
                #                 dict(type='HSigmoid', bias=3.0, divisor=6.0)))
                # else:
                #     se_cfg = None
                se_cfg = None
            
                stride = 2 if i == 0 else 1
                mid_channels = in_channels*self.expand_ratio
                layer = InvertedResidual(
                    in_channels=in_channels,
                    out_channels=w,
                    mid_channels=mid_channels,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    se_cfg=se_cfg,
                    with_expand_conv=(in_channels != mid_channels),
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp)
                
                in_channels = w
                block.append(layer)

            layers.append(block)
  

        # build the last two stage (3 and 4) (like MobileNet with efficientViT module)
        for w, d in zip(self.width_list[3:], self.depth_list[3:]):
            block = ModuleList()

            # if with_se:
            #     se_cfg = dict(
            #         channels=mid_channels,
            #         ratio=4,
            #         act_cfg=(dict(type='ReLU'),
            #                 dict(type='HSigmoid', bias=3.0, divisor=6.0)))
            # else:
            #     se_cfg = None
            se_cfg = None

            mid_channels = in_channels*self.expand_ratio
            layer = InvertedResidualV3_bn(
                in_channels=in_channels,
                out_channels=w,
                mid_channels=mid_channels,
                kernel_size=self.kernel_size,
                stride=2,
                se_cfg=se_cfg,
                with_expand_conv=(in_channels != mid_channels),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp)
            
            in_channels = w
            block.append(layer)

            for _ in range(d):
                # if with_se:
                #     se_cfg = dict(
                #         channels=mid_channels,
                #         ratio=4,
                #         act_cfg=(dict(type='ReLU'),
                #                 dict(type='HSigmoid', bias=3.0, divisor=6.0)))
                # else:
                #     se_cfg = None
                se_cfg = None

                mid_channels = in_channels*self.expand_ratio
                layer = EfficientViTBlock(
                    in_channels=in_channels,
                    out_channels=w,
                    kernel_size=self.kernel_size,
                    stride=1,
                    dim=self.dim,
                    se_cfg=se_cfg,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp)

                block.append(layer)

            layers.append(block)


        # # next, convert backbone MobileNetV3 to a semantic segmentation version
        # if self.arch == 'small':
        #     self.layer4.depthwise_conv.conv.stride = (1, 1)
        #     self.layer9.depthwise_conv.conv.stride = (1, 1)
        #     for i in range(4, len(layers)):
        #         layer = getattr(self, layers[i])
        #         if isinstance(layer, InvertedResidual):
        #             modified_module = layer.depthwise_conv.conv
        #         else:
        #             modified_module = layer.conv

        #         if i < 9:
        #             modified_module.dilation = (2, 2)
        #             pad = 2
        #         else:
        #             modified_module.dilation = (4, 4)
        #             pad = 4

        #         if not isinstance(modified_module, Conv2dAdaptivePadding):
        #             # Adjust padding
        #             pad *= (modified_module.kernel_size[0] - 1) // 2
        #             modified_module.padding = (pad, pad)
        # else:
        #     self.layer7.depthwise_conv.conv.stride = (1, 1)
        #     self.layer13.depthwise_conv.conv.stride = (1, 1)
        #     for i in range(7, len(layers)):
        #         layer = getattr(self, layers[i])
        #         if isinstance(layer, InvertedResidual):
        #             modified_module = layer.depthwise_conv.conv
        #         else:
        #             modified_module = layer.conv

        #         if i < 13:
        #             modified_module.dilation = (2, 2)
        #             pad = 2
        #         else:
        #             modified_module.dilation = (4, 4)
        #             pad = 4

        #         if not isinstance(modified_module, Conv2dAdaptivePadding):
        #             # Adjust padding
        #             pad *= (modified_module.kernel_size[0] - 1) // 2
        #             modified_module.padding = (pad, pad)

        return layers

    # def _make_layer(self):
    #     layers = []

    #     # build stem layer
    #     layer = ConvModule(
    #         in_channels=3,
    #         out_channels=self.width_list[0],
    #         kernel_size=self.kernel_size,
    #         stride=2,
    #         padding=1,
    #         conv_cfg=dict(type='Conv2dAdaptivePadding'),
    #         norm_cfg=self.norm_cfg,
    #         act_cfg=self.act_cfg)
    #     self.add_module('stem0', layer)
    #     layers.append('stem0')

    #     for i in range(self.depth_list[0]):
    #         layer = InvertedResidual(
    #             in_channels=self.width_list[0],
    #             out_channels=self.width_list[0],
    #             mid_channels=self.width_list[0],
    #             kernel_size=self.kernel_size,
    #             stride=1,
    #             se_cfg=None,
    #             with_expand_conv=False,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg,
    #             with_cp=self.with_cp)
    #         layer_name = f'stem{i + 1}'
    #         self.add_module(layer_name, layer)
    #         layers.append(layer_name)
    #     in_channels = self.width_list[0]

    #     # build the first two stage (1 and 2) (like MobileNet without efficientViT module)
    #     count = 0
    #     for w, d in zip(self.width_list[1:3], self.depth_list[1:3]):
    #         for i in range(d):
    #             # if with_se:
    #             #     se_cfg = dict(
    #             #         channels=mid_channels,
    #             #         ratio=4,
    #             #         act_cfg=(dict(type='ReLU'),
    #             #                 dict(type='HSigmoid', bias=3.0, divisor=6.0)))
    #             # else:
    #             #     se_cfg = None
    #             se_cfg = None
            
    #             stride = 2 if i == 0 else 1
    #             mid_channels = in_channels*self.expand_ratio
    #             layer = InvertedResidual(
    #                 in_channels=in_channels,
    #                 out_channels=w,
    #                 mid_channels=mid_channels,
    #                 kernel_size=self.kernel_size,
    #                 stride=stride,
    #                 se_cfg=se_cfg,
    #                 with_expand_conv=(in_channels != mid_channels),
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 with_cp=self.with_cp)
                
    #             in_channels = w
    #             layer_name = f'layer{count + 1}'
    #             self.add_module(layer_name, layer)
    #             layers.append(layer_name)
    #             count += 1

    #     # build the last two stage (3 and 4) (like MobileNet with efficientViT module)
    #     for w, d in zip(self.width_list[3:], self.depth_list[3:]):
    #         # if with_se:
    #         #     se_cfg = dict(
    #         #         channels=mid_channels,
    #         #         ratio=4,
    #         #         act_cfg=(dict(type='ReLU'),
    #         #                 dict(type='HSigmoid', bias=3.0, divisor=6.0)))
    #         # else:
    #         #     se_cfg = None
    #         se_cfg = None

    #         mid_channels = in_channels*self.expand_ratio
    #         layer = InvertedResidualV3_bn(
    #             in_channels=in_channels,
    #             out_channels=w,
    #             mid_channels=mid_channels,
    #             kernel_size=self.kernel_size,
    #             stride=2,
    #             se_cfg=se_cfg,
    #             with_expand_conv=(in_channels != mid_channels),
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg,
    #             with_cp=self.with_cp)
            
    #         in_channels = w
    #         layer_name = f'layer{count + 1}'
    #         self.add_module(layer_name, layer)
    #         layers.append(layer_name)
    #         count += 1

    #         for _ in range(d):
    #             # if with_se:
    #             #     se_cfg = dict(
    #             #         channels=mid_channels,
    #             #         ratio=4,
    #             #         act_cfg=(dict(type='ReLU'),
    #             #                 dict(type='HSigmoid', bias=3.0, divisor=6.0)))
    #             # else:
    #             #     se_cfg = None
    #             se_cfg = None

    #             mid_channels = in_channels*self.expand_ratio
    #             layer = EfficientViTBlock(
    #                 in_channels=in_channels,
    #                 out_channels=w,
    #                 kernel_size=self.kernel_size,
    #                 stride=1,
    #                 dim=self.dim,
    #                 se_cfg=se_cfg,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 with_cp=self.with_cp)

    #             layer_name = f'layer{count + 1}'
    #             self.add_module(layer_name, layer)
    #             layers.append(layer_name)
    #             count += 1


    #     # # next, convert backbone MobileNetV3 to a semantic segmentation version
    #     # if self.arch == 'small':
    #     #     self.layer4.depthwise_conv.conv.stride = (1, 1)
    #     #     self.layer9.depthwise_conv.conv.stride = (1, 1)
    #     #     for i in range(4, len(layers)):
    #     #         layer = getattr(self, layers[i])
    #     #         if isinstance(layer, InvertedResidual):
    #     #             modified_module = layer.depthwise_conv.conv
    #     #         else:
    #     #             modified_module = layer.conv

    #     #         if i < 9:
    #     #             modified_module.dilation = (2, 2)
    #     #             pad = 2
    #     #         else:
    #     #             modified_module.dilation = (4, 4)
    #     #             pad = 4

    #     #         if not isinstance(modified_module, Conv2dAdaptivePadding):
    #     #             # Adjust padding
    #     #             pad *= (modified_module.kernel_size[0] - 1) // 2
    #     #             modified_module.padding = (pad, pad)
    #     # else:
    #     #     self.layer7.depthwise_conv.conv.stride = (1, 1)
    #     #     self.layer13.depthwise_conv.conv.stride = (1, 1)
    #     #     for i in range(7, len(layers)):
    #     #         layer = getattr(self, layers[i])
    #     #         if isinstance(layer, InvertedResidual):
    #     #             modified_module = layer.depthwise_conv.conv
    #     #         else:
    #     #             modified_module = layer.conv

    #     #         if i < 13:
    #     #             modified_module.dilation = (2, 2)
    #     #             pad = 2
    #     #         else:
    #     #             modified_module.dilation = (4, 4)
    #     #             pad = 4

    #     #         if not isinstance(modified_module, Conv2dAdaptivePadding):
    #     #             # Adjust padding
    #     #             pad *= (modified_module.kernel_size[0] - 1) // 2
    #     #             modified_module.padding = (pad, pad)

    #     return layers

    # def forward(self, x):
    #     outs = []
    #     for i, layer_name in enumerate(self.layers):
    #         layer = getattr(self, layer_name)
    #         x = layer(x)
    #         if i in self.out_indices:
    #             outs.append(x)
    #     return outs
    
    def forward(self, x):
        outs = []
        x = self.layers[0][0](x) #the first conv of stem
        for i, block in enumerate(self.layers[0][1]): #stem block
            x = block(x)

        for i, layer in enumerate(self.layers[1:]): #main block
            for block in layer: #stem block
                x = block(x)
            if i in self.out_indices:
                outs.append(x)

        return outs

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
