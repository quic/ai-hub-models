# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch.nn as nn


###########################################
# HeadBlock, DownsampleBlock, NormalBlock, and EmbedBlock
##########################################
# HeadBlock
class HeadBlock(nn.Module):
    def __init__(self, chan, group_size=32, activ_type="prelu"):
        super().__init__()
        self.conv = Conv2dBlock(
            1, chan, 3, padding=1, stride=1, group=1, norm="bn", activ=activ_type
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

        if activ_type == "prelu":
            self.activ = nn.PReLU()
        elif activ_type == "relu":
            self.activ = nn.ReLU()
        else:
            self.activ = None
            assert 0, f"Unsupported activation function: {activ_type}"

    def forward(self, x):
        x = self.conv(x)
        x = self.bran1(x) + self.bran2(x)
        if self.activ:
            x = self.activ(x)
        return x


# DownsampleBlock
class DownsampleBlock(nn.Module):
    def __init__(self, chan, group_size=32, activ_type="prelu"):
        super().__init__()
        assert (
            chan % group_size == 0
        ), f"chan {chan:d} cannot be divided by group_size {group_size:d}"

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

        if activ_type == "prelu":
            self.activ = nn.PReLU()
        elif activ_type == "relu":
            self.activ = nn.ReLU()
        else:
            self.activ = None
            assert 0, f"Unsupported activation function: {activ_type}"

    def forward(self, x):
        x = self.bran1(x) + self.bran2(x)
        if self.activ:
            x = self.activ(x)
        return x


# NormalBlock
class NormalBlock(nn.Module):
    def __init__(self, chan, group_size=32, activ_type="prelu"):
        super().__init__()
        assert (
            chan % group_size == 0
        ), f"chan {chan:d} cannot be divided by group_size {group_size:d}"
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

        if activ_type == "prelu":
            self.activ = nn.PReLU()
        elif activ_type == "relu":
            self.activ = nn.ReLU()
        else:
            self.activ = None
            assert 0, f"Unsupported activation function: {activ_type}"

    def forward(self, x):
        x = self.model(x) + x
        if self.activ:
            x = self.activ(x)
        return x


class EmbedBlock(nn.Module):
    def __init__(self, chan, group_size=32, activ_type="prelu"):
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

    def forward(self, x):
        x = self.model(x)
        return x.flatten(start_dim=1)


###########################################
# Basic Blocks
##########################################
class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        padding=0,
        stride=1,
        group=1,
        norm="none",
        activ="none",
    ):
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

        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_chan)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

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

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x
