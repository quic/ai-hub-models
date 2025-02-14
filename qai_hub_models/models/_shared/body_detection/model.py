# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import math
from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn


def make_divisible(x: int, divisor: int) -> int:
    """
    Compute the closest number that is larger or equal to X and is divisible by DIVISOR.

    Inputs:
        x: int
            Input intenger
        divisor: int
            Divisor for the input number.
    Outputs: int
        Closest number that is larger or equal to X and is divisible by DIVISOR.
    """
    return math.ceil(x / divisor) * divisor


class Concat(nn.Module):
    """Tensor concatenation module"""

    def __init__(self, dimension: int = 1) -> None:
        """
        Inputs:
            dimension: int
                Dimension to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Inputs:
            x: list[torch.Tensor]
                List of tensors to be concatenated.
        Output: torch.Tensor
            Concatenated tensor.
        """
        return torch.cat(x, self.d)


def autopad(
    kernel_size: Union[int, tuple[int, int]],
    p: Optional[Union[int, tuple[int, int]]] = None,
) -> Union[int, tuple[int, int]]:
    """
    Compute padding size from kernel size.

    Input:
        kernel_size: int
            Kernel size.
        p: int | tuple[int, int]
            Padding size.
    Outputs: int
        Padding size
    """
    if p is not None:
        return p
    if isinstance(kernel_size, int):
        return kernel_size // 2
    assert 2 == len(kernel_size)
    # pyright and mypy complain that we're returning tuple[int, ...], but we've just asserted
    # that it's length two so we should be safe to ignore the error.
    return tuple([x // 2 for x in kernel_size])  # type: ignore[return-value]


class FusedConvBatchNorm(nn.Module):
    """Module of convolution, batch nornamlization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding=None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        """
        Initialize FusedConvBatchNorm.

        Inputs:
            in_channels: int
                Input channels.
            out_channels: int
                Output channels.
            kernel_size: int
                Kernel size.
            stride: int
                Convolution stride.
            groups: int
                Groups of channels for convolution.
            act: bool
                Whether to enable ReLU activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = (
            nn.ReLU(True)
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of FusedConvBatchNorm module.

        Inputs:
            x: torch.Tensor
                Input tensor
        Output: torch.Tensor
            Output tensor.
        """
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Bottleneck block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expand_ratio: float = 0.5,
    ) -> None:
        """
        Initialize Bottleneck module.

        Inputs:
            in_channels: int
                Input channels.
            out_channels: int
                Output channels.
            shortcut: bool
                Whether to enable shortcut connection.
            groups: int
                Groups of channels for convolution.
            expand_ratio: float
                Expand ratio of input channels to hidden channels.
        """
        super().__init__()
        hidden_channels = int(out_channels * expand_ratio)
        self.cv1 = FusedConvBatchNorm(in_channels, hidden_channels, 1, 1)
        self.cv2 = FusedConvBatchNorm(
            hidden_channels, out_channels, 3, 1, groups=groups
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of Bottleneck module.

        Inputs:
            x: torch.Tensor
                Input tensor.
        Outputs: torch.Tensor.
            Output tensor.
        """
        y = self.cv2(self.cv1(x))
        if self.add:
            y += x
        return y


class C3(nn.Module):
    """C3 block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        group: int = 1,
        expand_ratio: float = 0.5,
    ) -> None:
        """
        Initialize C3 module.

        Inputs:
            in_channels: int
                Input channels.
            out_channels: int
                Output channels.
            num_blocks: int
                Number of Bottleneck blocks.
            shortcut: bool
                Whether to enable shortcut connection.
            groups: int
                Groups of channels for convolution.
            expand_ratio: float
                Expand ratio of input channels to hidden channels.
        """
        super().__init__()
        hidden_channels = int(out_channels * expand_ratio)
        self.cv1 = FusedConvBatchNorm(in_channels, hidden_channels, 1, 1)
        self.cv2 = FusedConvBatchNorm(in_channels, hidden_channels, 1, 1)
        self.cv3 = FusedConvBatchNorm(2 * hidden_channels, out_channels, 1)
        self.m = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels, hidden_channels, shortcut, group, expand_ratio=1.0
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of C3 module.

        Inputs:
            x: torch.Tensor.
                Input tensor.
        Outputs: torch.Tensor.
            Output tensor
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> None:
        """
        Initialize SPPF module.

        Input:
            in_channels: int
                Input channels.
            out_channels: int
                Output channels.
            kernel_size: int
                Kernel size.
        """
        super().__init__()
        hiddel_channels = in_channels // 2
        self.cv1 = FusedConvBatchNorm(in_channels, hiddel_channels, 1, 1)
        self.cv2 = FusedConvBatchNorm(hiddel_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of SPPF module.

        Inputs:
            x: torch.Tensor.
                Input tensor.
        Outputs: torch.Tensor.
            Output tensor
        """
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detect(nn.Module):
    """Detector head module"""

    def __init__(
        self, num_classes: int = 80, anchors: tuple = (), ch: tuple = ()
    ) -> None:
        """
        Initialize Detector module.

        Inputs:
            num_classes: int
                Number of object classes.
            anchors: tuple
                Tuple of anchor sizes.
            ch: tuple
                Input channels of multi-scales
            inplace: bool
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_output = num_classes + 5
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.num_layers
        self.anchor_grid = [torch.zeros(1)] * self.num_layers
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        )
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.num_output * self.num_anchors, 1) for x in ch
        )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward computation of Detect module.

        Inputs:
            x: list[torch.Tensor].
                Input lisr of tensors.
        Outputs: list[torch.Tensor].
            Output list of tensors.
        """
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])
        return x


def parse_model(cfg: dict, ch: list[int]):
    """
    Generate model module from model configuration.

    Inputs:
        cfg: dict
            Model configurations.
        ch: list
            Input channels.
    Output:
        model: nn.Sequential
            Model layers.
        save: list
            List of layer indices that needs to be saved.
    """
    anchors, nc, gd, gw = (
        cfg["anchors"],
        cfg["nc"],
        cfg["depth_multiple"],
        cfg["width_multiple"],
    )
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    num_outputs = num_anchors * (nc + 5)
    layers = []
    save: list[int] = []
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate(cfg["backbone"] + cfg["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n
        if m in [FusedConvBatchNorm, Bottleneck, SPPF, C3, DoubleBlazeBlock]:
            c1, c2 = ch[f], args[0]
            if c2 != num_outputs:
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = (i, f, t, np)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class DoubleBlazeBlock(nn.Module):
    """
    DoubleBlaze block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 5,
        bias: bool = False,
    ) -> None:
        """
        Initialize DoubleBlaze block.

        Inputs:
            in_channels: int
                Number of input channels.
            out_channels: int
                Number of output channels.
            stride: int
                Convolution stride.
            kernel_size: int
                Kernel size.
            bias: bool.
                Enable bias in convolution.
        """
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_pooling = self.stride != 1
        self.channel_pad = out_channels - in_channels
        if self.channel_pad != 0:
            self.pad = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        padding = (kernel_size - 1) // 2
        hidden_channels = max(out_channels, in_channels) // 2

        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(in_channels),
            # pw-linear
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(hidden_channels),
        )
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=hidden_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(hidden_channels),
            # pw-linear
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of DoubleBlaze block.

        Input:
            x: torch.Tensor.
                Input tensor
        Output: torch.Tensor
            Output tensor.
        """
        h = self.conv1(x)
        h = self.conv2(h)

        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad != 0:
            x = self.pad(x)
        return self.act(h + x)


class Model(nn.Module):
    """Person/face detection model"""

    def __init__(self, model_cfg: dict, ch: int = 3) -> None:
        """
        Initialize person/face detection model.

        Inputs:
            ch: int
                Input channels.
            model_cfg: dict
                Model configuration
        """
        super().__init__()
        self.model, self.save = parse_model(deepcopy(model_cfg), ch=[ch])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward computation of Model.

        Inputs:
            x: torch.Tensor.
                Input image.
        Outputs: list[torch.Tensor]
            Multi-scale object detection output.
        """
        y: list[Optional[int]] = []
        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
