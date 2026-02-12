# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import init


class ConvBlock(nn.Module):
    """Standard convolution block with Batch normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        bn_eps: float = 1e-5,
        activation: Callable[[], nn.Module] | None = (lambda: nn.ReLU(inplace=True)),
    ) -> None:
        """
        Initialize a ConvBlock.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Convolution window size.
        stride
            Strides of the convolution.
        padding
            Padding value for convolution layer.
        dilation
            Dilation value for convolution layer.
        groups
            Number of groups.
        bias
            Whether the layer uses a bias vector.
        use_bn
            Whether to use BatchNorm layer.
        bn_eps
            Small float added to variance in Batch norm.
        activation
            Activation function or name of activation function.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    bias: bool = False,
    use_bn: bool = True,
    bn_eps: float = 1e-5,
    activation: Callable[[], nn.Module] | None = (lambda: nn.ReLU(inplace=True)),
) -> ConvBlock:
    """
    1x1 version of the standard convolution block.

    Parameters
    ----------
    in_channels
        Number of input channels.
    out_channels
        Number of output channels.
    stride
        Strides of the convolution.
    padding
        Padding value for convolution layer.
    groups
        Number of groups.
    bias
        Whether the layer uses a bias vector.
    use_bn
        Whether to use BatchNorm layer.
    bn_eps
        Small float added to variance in Batch norm.
    activation
        Activation function or name of activation function.

    Returns
    -------
    ConvBlock
        The 1x1 convolution block.
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
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
    use_bn: bool = True,
    bn_eps: float = 1e-5,
    activation: Callable[[], nn.Module] | None = (lambda: nn.ReLU(inplace=True)),
) -> ConvBlock:
    """
    3x3 version of the standard convolution block.

    Parameters
    ----------
    in_channels
        Number of input channels.
    out_channels
        Number of output channels.
    stride
        Strides of the convolution.
    padding
        Padding value for convolution layer.
    dilation
        Dilation value for convolution layer.
    groups
        Number of groups.
    bias
        Whether the layer uses a bias vector.
    use_bn
        Whether to use BatchNorm layer.
    bn_eps
        Small float added to variance in Batch norm.
    activation
        Activation function or name of activation function.

    Returns
    -------
    ConvBlock
        The 3x3 convolution block.
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
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 3,
    bias: bool = False,
    use_bn: bool = True,
    activation: Callable[[], nn.Module] | None = (lambda: nn.ReLU(inplace=True)),
) -> ConvBlock:
    """
    7x7 version of the standard convolution block.

    Parameters
    ----------
    in_channels
        Number of input channels.
    out_channels
        Number of output channels.
    stride
        Strides of the convolution.
    padding
        Padding value for convolution layer.
    bias
        Whether the layer uses a bias vector.
    use_bn
        Whether to use BatchNorm layer.
    activation
        Activation function or name of activation function.

    Returns
    -------
    ConvBlock
        The 7x7 convolution block.
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
    """Simple ResNet block for residual path in ResNet unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        bias: bool = False,
        use_bn: bool = True,
    ) -> None:
        """
        Initialize a ResBlock.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        stride
            Strides of the convolution.
        bias
            Whether the layer uses a bias vector.
        use_bn
            Whether to use BatchNorm layer.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(x)


class ResBottleneck(nn.Module):
    """ResNet bottleneck block for residual path in ResNet unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        padding: int = 1,
        dilation: int = 1,
        conv1_stride: bool = False,
        bottleneck_factor: int = 4,
    ) -> None:
        """
        Initialize a ResBottleneck.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        stride
            Strides of the convolution.
        padding
            Padding value for the second convolution layer.
        dilation
            Dilation value for the second convolution layer.
        conv1_stride
            Whether to use stride in the first or the second convolution layer of the block.
        bottleneck_factor
            Bottleneck factor.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class ResUnit(nn.Module):
    """ResNet unit with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        bottleneck: bool = True,
        conv1_stride: bool = False,
    ) -> None:
        """
        Initialize a ResUnit.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        stride
            Strides of the convolution.
        padding
            Padding value for the second convolution layer in bottleneck.
        dilation
            Dilation value for the second convolution layer in bottleneck.
        bias
            Whether the layer uses a bias vector.
        use_bn
            Whether to use BatchNorm layer.
        bottleneck
            Whether to use a bottleneck or simple block in units.
        conv1_stride
            Whether to use stride in the first or the second convolution layer of the block.
        """
        super().__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.body: nn.Module
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x) if self.resize_identity else x
        x = self.body(x)
        x = x + identity
        return self.activ(x)


class ResInitBlock(nn.Module):
    """ResNet specific initial block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize a ResInitBlock.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        """
        super().__init__()
        self.conv = conv7x7_block(
            in_channels=in_channels, out_channels=out_channels, stride=2
        )

        self.pool = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(3, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.pool(x)


class ResNet(nn.Module):
    """ResNet model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385."""

    def __init__(
        self,
        channels: list[list[int]],
        init_block_channels: int,
        bottleneck: bool,
        conv1_stride: bool,
        in_channels: int = 3,
        in_size: tuple[int, int] = (128, 128),
        num_classes: int = 265,
    ) -> None:
        """
        Initialize a ResNet.

        Parameters
        ----------
        channels
            Number of output channels for each unit.
        init_block_channels
            Number of output channels for the initial unit.
        bottleneck
            Whether to use a bottleneck or simple block in units.
        conv1_stride
            Whether to use stride in the first or the second convolution layer in units.
        in_channels
            Number of input channels.
        in_size
            Spatial size of the expected input image.
        num_classes
            Number of classification classes.
        """
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

    def _init_params(self) -> None:
        for _name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rgb2gray_block(x)
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        return self.output(feature)


def get_resnet(
    blocks: int,
    bottleneck: bool | None = None,
    conv1_stride: bool = True,
    width_scale: float = 1.0,
    model_name: str | None = None,
    pretrained: bool = False,
    root: str = os.path.join("~", ".torch", "models"),
    **kwargs: Any,
) -> ResNet:
    """
    Create ResNet model with specific parameters.

    Parameters
    ----------
    blocks
        Number of blocks.
    bottleneck
        Whether to use a bottleneck or simple block in units.
    conv1_stride
        Whether to use stride in the first or the second convolution layer in units.
    width_scale
        Scale factor for width of layers.
    model_name
        Model name for loading pretrained model.
    pretrained
        Whether to load the pretrained weights for model.
    root
        Location for keeping the model parameters.
    **kwargs
        Additional keyword arguments for ResNet constructor.

    Returns
    -------
    ResNet
        The ResNet model instance.
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

    channels = [
        [ci] * li for (ci, li) in zip(channels_per_layers, layers, strict=False)
    ]

    if width_scale != 1.0:
        channels = [
            [
                (
                    int(cij * width_scale)
                    if (i != len(channels) - 1) or (j != len(ci) - 1)
                    else cij
                )
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


def resnet18_wd2(**kwargs: bool | str | None) -> ResNet:
    """
    ResNet-18 model with 0.5 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to get_resnet, including:
        - pretrained: Whether to load the pretrained weights for model.
        - root: Location for keeping the model parameters.

    Returns
    -------
    ResNet
        The ResNet-18 model with 0.5 width scale.
    """
    return get_resnet(blocks=18, width_scale=0.5, model_name="resnet18_wd2", **kwargs)  # type: ignore[arg-type]
