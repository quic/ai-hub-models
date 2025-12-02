# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from torch import Tensor, nn


###########################################
# HeadBlock, DownsampleBlock, NormalBlock, and EmbedBlock
##########################################
# HeadBlock
class HeadBlock(nn.Module):
    def __init__(self, chan: int, group_size: int = 32, activ_type: str = "prelu"):
        """
        Initialize the HeadBlock module for feature extraction.

        Parameters
        ----------
        chan : int
            Number of channels for the convolutional layers.

        group_size : int, optional
            Group size for grouped convolutions. Defaults to 32.

        activ_type : str, optional
            Activation function type ('prelu' or 'relu'). Defaults to 'prelu'.
        """
        super().__init__()
        self.conv = Conv2dBlock(
            3, chan, 3, padding=1, stride=1, group=1, norm="bn", activ=activ_type
        )
        bran1_block = [
            Conv2dBlock(
                chan,
                chan,
                3,
                padding=1,
                stride=2,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                chan, chan, 1, padding=0, stride=1, group=1, norm="bn", activ=activ_type
            ),
            Conv2dBlock(
                chan,
                chan,
                3,
                padding=1,
                stride=1,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                chan, chan, 1, padding=0, stride=1, group=1, norm="bn", activ="none"
            ),
        ]

        self.bran1 = nn.Sequential(*bran1_block)

        bran2_block = [
            Conv2dBlock(
                chan,
                chan,
                3,
                padding=1,
                stride=2,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                chan, chan, 1, padding=0, stride=1, group=1, norm="bn", activ="none"
            ),
        ]
        self.bran2 = nn.Sequential(*bran2_block)
        self.activ: nn.Module | None
        if activ_type == "prelu":
            self.activ = nn.PReLU()
        elif activ_type == "relu":
            self.activ = nn.ReLU()
        else:
            self.activ = None
            raise ValueError(f"Unsupported activation function: {activ_type}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward pass through the HeadBlock.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [N, 3, H, W].

        Returns
        -------
        Tensor
            Output tensor of shape [N, chan, H/2, W/2].
        """
        x = self.conv(x)
        x = self.bran1(x) + self.bran2(x)
        if self.activ:
            x = self.activ(x)
        return x


# DownsampleBlock
class DownsampleBlock(nn.Module):
    def __init__(self, chan: int, group_size: int = 32, activ_type: str = "prelu"):
        """
        Initialize the DownsampleBlock module for reducing spatial dimensions while increasing channel depth.

        Parameters
        ----------
        chan : int
            Number of input channels.

        group_size : int, optional
            Group size for grouped convolutions. Defaults to 32.

        activ_type : str, optional
            Activation function type ('prelu' or 'relu'). Defaults to 'prelu'.

        """
        super().__init__()
        assert chan % group_size == 0, (
            f"chan {chan:d} cannot be divided by group_size {group_size:d}"
        )

        self.bran1 = nn.Sequential(
            Conv2dBlock(
                chan,
                2 * chan,
                3,
                padding=1,
                stride=2,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                2 * chan,
                2 * chan,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            ),
        )

        bran2_block = [
            Conv2dBlock(
                chan,
                4 * chan,
                3,
                padding=1,
                stride=2,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                4 * chan,
                2 * chan,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            ),
            Conv2dBlock(
                2 * chan,
                4 * chan,
                3,
                padding=1,
                stride=1,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                4 * chan,
                2 * chan,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ="none",
            ),
        ]

        self.bran2 = nn.Sequential(*bran2_block)
        self.activ: nn.Module | None
        if activ_type == "prelu":
            self.activ = nn.PReLU()
        elif activ_type == "relu":
            self.activ = nn.ReLU()
        else:
            self.activ = None
            assert 0, f"Unsupported activation function: {activ_type}"

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward pass through the DownsampleBlock.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [N, chan, H, W].

        Returns
        -------
        Tensor
            Output tensor of shape [N, 2*chan, H/2, W/2].
        """
        x = self.bran1(x) + self.bran2(x)
        if self.activ:
            x = self.activ(x)
        return x


# NormalBlock
class NormalBlock(nn.Module):
    def __init__(self, chan: int, group_size: int = 32, activ_type: str = "prelu"):
        """
        Initialize the NormalBlock module for standard feature extraction without changing spatial dimensions.

        Parameters
        ----------
        chan : int
            Number of input and output channels.

        group_size : int, optional
            Group size for grouped convolutions. Defaults to 32.

        activ_type : str, optional
            Activation function type ('prelu' or 'relu'). Defaults to 'prelu'.

        """
        super().__init__()
        assert chan % group_size == 0, (
            f"chan {chan:d} cannot be divided by group_size {group_size:d}"
        )
        model_block = [
            Conv2dBlock(
                chan,
                2 * chan,
                3,
                padding=1,
                stride=1,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                2 * chan,
                chan,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ="prelu",
            ),
            Conv2dBlock(
                chan,
                2 * chan,
                3,
                padding=1,
                stride=1,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                2 * chan, chan, 1, padding=0, stride=1, group=1, norm="bn", activ="none"
            ),
        ]

        self.model = nn.Sequential(*model_block)
        self.activ: nn.Module | None
        if activ_type == "prelu":
            self.activ = nn.PReLU()
        elif activ_type == "relu":
            self.activ = nn.ReLU()
        else:
            self.activ = None
            assert 0, f"Unsupported activation function: {activ_type}"

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward pass through the NormalBlock.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [N, chan, H, W].

        Returns
        -------
        Tensor
            Output tensor of shape [N, chan, H, W] (residual connection maintains spatial dimensions).
        """
        x = self.model(x) + x
        if self.activ:
            x = self.activ(x)
        return x


class EmbedBlock(nn.Module):
    def __init__(self, chan: int, group_size: int = 32, activ_type: str = "prelu"):
        """
        Initialize the EmbedBlock module for converting feature maps into flattened embeddings.

        Parameters
        ----------
        chan : int
            Number of input and output channels.

        group_size : int, optional
            Group size for grouped convolutions. Defaults to 32.

        activ_type : str, optional
            Activation function type ('prelu' or 'relu'). Defaults to 'prelu'.
        """
        super().__init__()
        self.model = nn.Sequential(
            Conv2dBlock(
                chan,
                chan,
                8,
                padding=0,
                stride=1,
                group=chan // group_size,
                norm="none",
                activ="none",
            ),
            Conv2dBlock(
                chan, chan, 1, padding=0, stride=1, group=1, norm="bn", activ=activ_type
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward pass through the EmbedBlock.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [N, chan, H, W] where H,W >= 8.

        Returns
        -------
        Tensor
            Flattened output tensor of shape [N, chan * (H-7) * (W-7)].
        """
        x = self.model(x)
        return x.flatten(start_dim=1)


###########################################
# Basic Blocks
##########################################
class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int | tuple,
        padding: int | tuple = 0,
        stride: int | tuple = 1,
        group: int = 1,
        norm: str = "none",
        activ: str = "none",
    ):
        """
        Initialize the Conv2dBlock module with convolution, normalization, and activation layers.

        Parameters
        ----------
        in_chan : int
            Number of input channels.

        out_chan : int
            Number of output channels.

        kernel_size : int or tuple
            Size of the convolutional kernel.

        padding : int or tuple, optional
            Zero-padding added to both sides of the input. Defaults to 0.

        stride : int or tuple, optional
            Stride of the convolution. Defaults to 1.

        group : int, optional
            Number of blocked connections from input channels to output channels. Defaults to 1.

        norm : str, optional
            Normalization type ('bn' for BatchNorm2d, 'none'). Defaults to 'none'.

        activ : str, optional
            Activation function type ('prelu', 'relu', 'sigmoid', 'none'). Defaults to 'none'.

        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=group,
            bias=False,
        )

        self.norm: nn.BatchNorm2d | None
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_chan)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        self.activ: nn.Module | None
        if activ == "prelu":
            self.activ = nn.PReLU()
        elif activ == "relu":
            self.activ = nn.ReLU()
        elif activ == "sigmoid":
            self.activ = nn.Sigmoid()
        elif activ == "none":
            self.activ = None
        else:
            assert 0, f"Unsupported activation layer: {activ}"

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward pass through the Conv2dBlock.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [N, in_chan, H, W].

        Returns
        -------
        Tensor
            Output tensor of shape [N, out_chan, H_out, W_out], where H_out and W_out depend on kernel_size, padding, and stride.
        """
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x
