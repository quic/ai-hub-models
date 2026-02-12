# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
import math
from typing import cast

import torch
from torch import nn


def make_divisible(x: int, divisor: int) -> int:
    """
    Compute the closest number that is larger or equal to x and is divisible by divisor.

    Parameters
    ----------
    x
        Input integer.
    divisor
        Divisor for the input number.

    Returns
    -------
    result : int
        Closest number that is larger or equal to x and is divisible by divisor.
    """
    return math.ceil(x / divisor) * divisor


class Concat(nn.Module):
    """Tensor concatenation module"""

    def __init__(self, dimension: int = 1) -> None:
        """
        Parameters
        ----------
        dimension
            Dimension to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            List of tensors to be concatenated.

        Returns
        -------
        output : torch.Tensor
            Concatenated tensor.
        """
        return torch.cat(x, self.d)


def autopad(
    kernel_size: int | tuple[int, int],
    p: int | tuple[int, int] | None = None,
) -> int | tuple[int, int]:
    """
    Compute padding size from kernel size.

    Parameters
    ----------
    kernel_size
        Kernel size.
    p
        Padding size. If None, computed from kernel_size.

    Returns
    -------
    padding : int | tuple[int, int]
        Padding size.
    """
    if p is not None:
        return p
    if isinstance(kernel_size, int):
        return kernel_size // 2
    assert len(kernel_size) == 2
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
        padding: int | tuple[int, int] | None = None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        """
        Initialize FusedConvBatchNorm.

        Parameters
        ----------
        in_channels
            Input channels.
        out_channels
            Output channels.
        kernel_size
            Kernel size.
        stride
            Convolution stride.
        padding
            Padding size. If None, computed from kernel_size.
        groups
            Groups of channels for convolution.
        act
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
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = (
            nn.ReLU(True)
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of FusedConvBatchNorm module.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        output : torch.Tensor
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

        Parameters
        ----------
        in_channels
            Input channels.
        out_channels
            Output channels.
        shortcut
            Whether to enable shortcut connection.
        groups
            Groups of channels for convolution.
        expand_ratio
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

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        output : torch.Tensor
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

        Parameters
        ----------
        in_channels
            Input channels.
        out_channels
            Output channels.
        num_blocks
            Number of Bottleneck blocks.
        shortcut
            Whether to enable shortcut connection.
        group
            Groups of channels for convolution.
        expand_ratio
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

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> None:
        """
        Initialize SPPF module.

        Parameters
        ----------
        in_channels
            Input channels.
        out_channels
            Output channels.
        kernel_size
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

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
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

        Parameters
        ----------
        num_classes
            Number of object classes.
        anchors
            Tuple of anchor sizes.
        ch
            Input channels of multi-scales.
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

        Parameters
        ----------
        x
            Input list of tensors.

        Returns
        -------
        output : list[torch.Tensor]
            Output list of tensors.
        """
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])
        return x


def build_gear_guard_net_model(
    cfg: dict[str, list], ch: list[int]
) -> tuple[nn.Sequential, list[int]]:
    """
    Generate model module from model configuration.

    Parameters
    ----------
    cfg
        Model configurations.
    ch
        Input channels.

    Returns
    -------
    model : nn.Sequential
        Model layers.
    save : list[int]
        List of layer indices that needs to be saved.
    """
    anchors, nc, gd, gw = (
        cfg["anchors"],
        cfg["nc"],
        cfg["depth_multiple"],
        cfg["width_multiple"],
    )
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    num_outputs = num_anchors * (cast(int, nc) + 5)
    layers = []
    save: list[int] = []
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate(cfg["backbone"] + cfg["head"]):
        m = eval(m) if isinstance(m, str) else m
        with contextlib.suppress(NameError):
            for j, a in enumerate(args):
                args[j] = eval(a) if isinstance(a, str) else a

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
    """DoubleBlaze block"""

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

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        stride
            Convolution stride.
        kernel_size
            Kernel size.
        bias
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
            nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.03),
            # pw-linear
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(hidden_channels, eps=0.001, momentum=0.03),
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
            nn.BatchNorm2d(hidden_channels, eps=0.001, momentum=0.03),
            # pw-linear
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
        )

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of DoubleBlaze block.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
        """
        h = self.conv1(x)
        h = self.conv2(h)

        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad != 0:
            x = self.pad(x)
        return self.act(h + x)
