# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.face_attrib_evaluator import FaceAttribNetEvaluator
from qai_hub_models.models.face_attrib_net.layers import (
    Conv2dBlock,
    DownsampleBlock,
    EmbedBlock,
    HeadBlock,
    NormalBlock,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "face_attrib_net"
MODEL_ASSET_VERSION = "1"
DEFAULT_WEIGHTS = "multitask_FR_state_dict.pt"

OUT_NAMES = [
    "id_feature",
    "liveness_feature",
    "eye_closeness",
    "glasses",
    "mask",
    "sunglasses",
]


class FaceAttribNet(BaseModel):
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

    def forward(self, image, target=None):
        """
        Run FaceAttribNet on cropped and pre-processed 128x128 face `image`, and produce various attributes.

        Parameters:
            image: Pixel values pre-processed
                   3-channel Color Space

        Returns:
            Face attributes vector
        """
        image = (image - 0.5) / 0.50196078
        x = self.head_converter(image)
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

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None):
        """Load from a weightfile"""
        faceattribnet_model = FaceAttribNet(
            64,
            [2, 6, 3],
            fea_only=True,
            liveness=True,
            openness=True,
            glasses=True,
            sunglasses=True,
            mask=True,
        )

        # "actual" because we completely ignore the method parameter
        actual_checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        )

        faceattribnet_model.load_state_dict(
            load_torch(actual_checkpoint_path)["model_state"]
        )
        faceattribnet_model.to(torch.device("cpu"))
        faceattribnet_model.eval()
        return faceattribnet_model

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return OUT_NAMES

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        return FaceAttribNetEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["face_attrib_dataset"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "face_attrib_dataset"
