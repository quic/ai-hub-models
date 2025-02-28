# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn


class FaceNet(nn.Module):
    def __init__(
        self,
        chan,
        blks_per_layer,
        fea_only=True,
        liveness=True,
        openness=True,
        glasses=True,
        mask=True,
        sunglasses=True,
        group_size=32,
        activ_type="prelu",
    ):
        super().__init__()

        self.head_converter = nn.Conv2d(
            3, 1, 1, stride=1, padding=0, groups=1, bias=False
        )
        self.chan = chan
        self.head = HeadBlock(chan)
        self.blks_per_layer = blks_per_layer
        self.fea_only = fea_only

        self.main_module = nn.ModuleList()
        for i in range(len(self.blks_per_layer)):
            self.main_module.append(self._make_net(self.chan, self.blks_per_layer[i]))
            self.chan *= 2

        self.embed = EmbedBlock(self.chan)

        self.liveness = liveness
        if self.liveness:
            self.base_chan = chan
            self.liveness_bran1_gconv = Conv2dBlock(
                self.base_chan * 4,
                self.base_chan * 2,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 4 // group_size,
                norm="bn",
                activ="none",
            )
            self.liveness_bran1_conv = Conv2dBlock(
                self.base_chan * 2,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.liveness_bran2_gconv = Conv2dBlock(
                self.base_chan * 10,
                self.base_chan * 5,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 10 // group_size,
                norm="bn",
                activ="none",
            )
            self.liveness_bran2_conv = Conv2dBlock(
                self.base_chan * 5,
                self.base_chan,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.liveness_fc = nn.Linear(self.base_chan * 4 * 4, self.base_chan // 2)

        self.openness = openness
        if self.openness:
            self.base_chan = chan
            self.openness_bran1_gconv = Conv2dBlock(
                self.base_chan * 4,
                self.base_chan * 2,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 4 // group_size,
                norm="none",
                activ="none",
            )
            self.openness_bran1_conv = Conv2dBlock(
                self.base_chan * 2,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.openness_bran2_gconv = Conv2dBlock(
                self.base_chan * 10,
                self.base_chan * 5,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 10 // group_size,
                norm="none",
                activ="none",
            )
            self.openness_bran2_conv = Conv2dBlock(
                self.base_chan * 5,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.openness_ave = nn.AvgPool2d(kernel_size=4, stride=1)
            self.openness_cls = nn.Linear(self.base_chan * 2, 2)

        self.glasses = glasses
        if self.glasses:
            self.base_chan = chan
            self.eyeglasses_bran1_gconv = Conv2dBlock(
                self.base_chan * 4,
                self.base_chan * 2,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 4 // group_size,
                norm="none",
                activ="none",
            )
            self.eyeglasses_bran1_conv = Conv2dBlock(
                self.base_chan * 2,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.eyeglasses_bran2_gconv = Conv2dBlock(
                self.base_chan * 10,
                self.base_chan * 5,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 10 // group_size,
                norm="none",
                activ="none",
            )
            self.eyeglasses_bran2_conv = Conv2dBlock(
                self.base_chan * 5,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.eyeglasses_ave = nn.AvgPool2d(kernel_size=4, stride=1)
            self.eyeglasses_cls = nn.Linear(self.base_chan * 2, 2)

        self.sunglasses = sunglasses
        if self.sunglasses:
            self.base_chan = chan
            self.sunglasses_bran1_gconv = Conv2dBlock(
                self.base_chan * 4,
                self.base_chan * 2,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 4 // group_size,
                norm="none",
                activ="none",
            )
            self.sunglasses_bran1_conv = Conv2dBlock(
                self.base_chan * 2,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.sunglasses_bran2_gconv = Conv2dBlock(
                self.base_chan * 10,
                self.base_chan * 5,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 10 // group_size,
                norm="none",
                activ="none",
            )
            self.sunglasses_bran2_conv = Conv2dBlock(
                self.base_chan * 5,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.sunglasses_ave = nn.AvgPool2d(kernel_size=4, stride=1)
            self.sunglasses_cls = nn.Linear(self.base_chan * 2, 2)
            self.sunglasses_softmax = nn.Softmax(dim=1)

        self.mask = mask
        if self.mask:
            self.base_chan = chan
            self.mask_bran1_gconv = Conv2dBlock(
                self.base_chan * 4,
                self.base_chan * 2,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 4 // group_size,
                norm="none",
                activ="none",
            )
            self.mask_bran1_conv = Conv2dBlock(
                self.base_chan * 2,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.mask_bran2_gconv = Conv2dBlock(
                self.base_chan * 10,
                self.base_chan * 5,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 10 // group_size,
                norm="none",
                activ="none",
            )
            self.mask_bran2_conv = Conv2dBlock(
                self.base_chan * 5,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.mask_ave = nn.AvgPool2d(kernel_size=4, stride=1)
            self.mask_cls = nn.Linear(self.base_chan * 2, 2)

    def _make_net(self, chan, n):
        cnn_x = []
        cnn_x += [DownsampleBlock(chan)]
        for i in range(n - 1):
            cnn_x += [NormalBlock(2 * chan)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x, target=None):
        x = self.head_converter(x)
        fea1 = self.head(x)

        fea2 = self.main_module[0](fea1)
        fea3 = self.main_module[1](fea2)
        fea4 = self.main_module[2](fea3)

        fr_out = self.embed(fea4)

        outputs = []
        outputs.append(fr_out)

        if self.liveness:
            a = self.liveness_bran1_conv(self.liveness_bran1_gconv(fea3))
            a = torch.cat((a, fea4), dim=1)
            a = self.liveness_bran2_conv(self.liveness_bran2_gconv(a))
            a = a.flatten(start_dim=1)
            a = self.liveness_fc(a)
            outputs.append(a)

        if self.openness:
            a = self.openness_bran1_conv(self.openness_bran1_gconv(fea3))
            a = torch.cat((a, fea4), dim=1)
            a = self.openness_ave(
                self.openness_bran2_conv(self.openness_bran2_gconv(a))
            )
            a = a.flatten(start_dim=1)
            a = self.openness_cls(a)
            outputs.append(a)

        if self.glasses:
            a = self.eyeglasses_bran1_gconv(fea3)
            a = self.eyeglasses_bran1_conv(a)
            a = torch.cat((a, fea4), dim=1)
            a = self.eyeglasses_ave(
                self.eyeglasses_bran2_conv(self.eyeglasses_bran2_gconv(a))
            )
            a = a.flatten(start_dim=1)
            a = self.eyeglasses_cls(a)
            outputs.append(a)

        if self.mask:
            a = self.mask_bran1_gconv(fea3)
            a = self.mask_bran1_conv(a)
            a = torch.cat((a, fea4), dim=1)
            a = self.mask_ave(self.mask_bran2_conv(self.mask_bran2_gconv(a)))
            a = a.flatten(start_dim=1)
            a = self.mask_cls(a)
            outputs.append(a)

        if self.sunglasses:
            a = self.sunglasses_bran1_conv(self.sunglasses_bran1_gconv(fea3))
            a = torch.cat((a, fea4), dim=1)
            a = self.sunglasses_ave(
                self.sunglasses_bran2_conv(self.sunglasses_bran2_gconv(a))
            )
            a = a.flatten(start_dim=1)
            a = self.sunglasses_cls(a)
            a = self.sunglasses_softmax(a)
            outputs.append(a)

        return outputs


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
