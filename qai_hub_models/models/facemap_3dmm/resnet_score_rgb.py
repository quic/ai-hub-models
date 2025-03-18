# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

import torch.nn as nn
import torch.nn.init as init


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        bias=False,
        use_bn=True,
        bn_eps=1e-5,
        activation=(lambda: nn.ReLU(inplace=True)),
    ):
        super().__init__()
        self.activate = activation is not None
        self.use_bn = use_bn
        self.use_pad = isinstance(padding, (list, tuple)) and (len(padding) == 4)

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(
    in_channels,
    out_channels,
    stride=1,
    padding=0,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True)),
):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
    )


def conv3x3_block(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation=(lambda: nn.ReLU(inplace=True)),
):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
    )


def conv7x7_block(
    in_channels,
    out_channels,
    stride=1,
    padding=3,
    bias=False,
    use_bn=True,
    activation=(lambda: nn.ReLU(inplace=True)),
):
    """
    7x7 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        use_bn=use_bn,
        activation=activation,
    )


class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    """

    def __init__(self, in_channels, out_channels, stride, bias=False, use_bn=True):
        super().__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            use_bn=use_bn,
        )
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=use_bn,
            activation=None,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(nn.Module):
    """
    ResNet bottleneck block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        padding=1,
        dilation=1,
        conv1_stride=False,
        bottleneck_factor=4,
    ):
        super().__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1),
        )
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride),
            padding=padding,
            dilation=dilation,
        )
        self.conv3 = conv1x1_block(
            in_channels=mid_channels, out_channels=out_channels, activation=None
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResUnit(nn.Module):
    """
    ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        padding=1,
        dilation=1,
        bias=False,
        use_bn=True,
        bottleneck=True,
        conv1_stride=False,
    ):
        super().__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv1_stride=conv1_stride,
            )
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                use_bn=use_bn,
            )
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                use_bn=use_bn,
                activation=None,
            )
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class ResInitBlock(nn.Module):
    """
    ResNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv7x7_block(
            in_channels=in_channels, out_channels=out_channels, stride=2
        )

        self.pool = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(3, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResNet(nn.Module):
    """
    ResNet model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(
        self,
        channels,
        init_block_channels,
        bottleneck,
        conv1_stride,
        in_channels=3,
        in_size=(128, 128),
        num_classes=265,
    ):
        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.rgb2gray_block = conv1x1_block(
            in_channels=in_channels,
            out_channels=1,
            use_bn=False,
            activation=None,
            bias=True,
        )
        self.features = nn.Sequential()
        self.features.add_module(
            "init_block", ResInitBlock(in_channels=1, out_channels=init_block_channels)
        )
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module(
                    f"unit{j + 1}",
                    ResUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        bottleneck=bottleneck,
                        conv1_stride=conv1_stride,
                    ),
                )
                in_channels = out_channels
            self.features.add_module(f"stage{i + 1}", stage)
        self.features.add_module("final_pool", nn.AvgPool2d(kernel_size=4, stride=1))

        # Total 264 outputs including 219 shape coefficients, 39 exp coefficients, 3 rotation and 3 camera intrinsic parameters
        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes,
        )

        self.sigmoid = nn.Sigmoid()

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.rgb2gray_block(x)
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        out = self.output(feature)

        return out


def get_resnet(
    blocks,
    bottleneck=None,
    conv1_stride=True,
    width_scale=1.0,
    model_name=None,
    pretrained=False,
    root=os.path.join("~", ".torch", "models"),
    **kwargs,
):
    """
    Create ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if bottleneck is None:
        bottleneck = blocks >= 50

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError(f"Unsupported ResNet with number of blocks: {blocks}")

    if bottleneck:
        assert sum(layers) * 3 + 2 == blocks
    else:
        assert sum(layers) * 2 + 2 == blocks

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [
            [
                int(cij * width_scale)
                if (i != len(channels) - 1) or (j != len(ci) - 1)
                else cij
                for j, cij in enumerate(ci)
            ]
            for i, ci in enumerate(channels)
        ]
        init_block_channels = int(init_block_channels * width_scale)

    net = ResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs,
    )

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError(
                "Parameter `model_name` should be properly initialized for loading pretrained model."
            )
        from .model_store import download_model

        download_model(net=net, model_name=model_name, local_model_store_dir_path=root)

    return net


def resnet18_wd2(**kwargs):
    """
    ResNet-18 model with 0.5 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, width_scale=0.5, model_name="resnet18_wd2", **kwargs)
