# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math

import torch
import torch.nn as nn

from qai_hub_models.models._shared.face_detection.layers import (
    Block3x3,
    CBAModule,
    DetectModule,
    HeadModule,
    SeModule,
    UpModule,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "face_det_lite"
MODEL_ASSET_VERSION = "1"
DEFAULT_WEIGHTS = "qfd360_sl_model.pt"


class FaceDetLite_model(BaseModel):
    """
    qualcomm face detector model.
    Detect bounding box for face,
    Detect landmarks: face landmarks.
    The output will be saved as 3 maps which will be decoded to final result in the FaceDetLite_App.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None):
        """Load FaceDetLite from a weightfile created by the source FaceDetLite repository."""

        checkpoint_to_load = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        )
        FaceDetLite_model = FaceDetLite()
        FaceDetLite_model.load_state_dict(load_torch(checkpoint_to_load)["model_state"])
        FaceDetLite_model.to(torch.device("cpu"))

        return cls(FaceDetLite_model)

    def forward(self, image):
        """
        Run FaceDetLite on `image`, and produce a the list of face bounding box

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   1-channel gray scale image

        Returns:
            heatmap: N,C,H,W the heatmap for the person/face detection.
            bbox: N,C*4, H,W the bounding box coordinate as a map.
            landmark: N,C*10,H,W the coordinates of landmarks as a map.
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"input": ((batch_size, 1, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["heatmap", "bbox", "landmark"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["input"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["heatmap", "bbox", "landmark"]


class FaceDetLite(nn.Module):
    def __init__(
        self,
        wide: int = 32,
        has_ext: bool = False,
        upmode: str = "UCBA",
        act: str = "relu",
        RGB: bool = False,
        has_se: bool = True,
        phase: str = "train",
    ):
        super().__init__()
        """
        FaceDetLite face detector model for face and landmark detection.
        output face bounding box and 5 landmarks.

        Parameters:
            wide: the channel size of bandwith of the intermediate layers
            has_ext: if add extension layer in the head module.
            upmode: upsampling mode.
            act: activation function.
            RGB: if the input is a 3 channel RGB
            has_se: if has the se module
            phase: "train" or "test"

        Returns:
            FaceDetLite model instance.
        """
        self.use_rgb = RGB
        self.has_landmark = True
        # define backbone
        self.bb = Mbv3SmallFast(act, RGB, has_se)

        c1, c2 = self.bb.uplayer_shape
        act = "relu"
        self.conv3 = CBAModule(
            self.bb.output_channels,
            wide,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            act=act,
        )  # s32
        self.connect1 = CBAModule(c1, wide, kernel_size=1, act=act)  # s8
        self.connect2 = CBAModule(c2, wide, kernel_size=1, act=act)  # s16

        self.up0 = UpModule(
            wide, wide, kernel_size=2, stride=2, mode=upmode, act=act
        )  # s16
        self.up1 = UpModule(
            wide, wide, kernel_size=2, stride=2, mode=upmode, act=act
        )  # s8
        self.detect = DetectModule(wide, act=act)

        self.center = HeadModule(wide, 1, act=act)
        self.box = HeadModule(wide, 4, act=act)

        if self.has_landmark:
            self.landmark = HeadModule(wide, 10, act=act)
        self.phase = phase

        self.bridge = nn.Conv2d(
            wide * 2, wide, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, input):
        """
        input: N,C,H,W (1,1,480,640) tensor of input image
        return: 3 tensors including
        heatmap: N,C,H,W (1,1,60,80)
        bbox: N,C,H,W (1,4,120,80)
        landmark: N,C,H,W (1,10,60,80)
        """

        s8_, s16_, s32_ = self.bb(input)
        s32 = self.conv3(s32_)

        s16 = self.up0(s32) + self.connect2(s16_)
        s8 = self.up1(s16) + self.connect1(s8_)
        x = self.detect(s8)  # s4: B,C,200,200

        center = self.center(x)
        box = self.box(x)

        if self.has_landmark:
            landmark = self.landmark(x)
            if self.phase == "test":
                return center.sigmoid(), box, landmark

            return center, box, landmark


class Mbv3SmallFast(nn.Module):
    def __init__(self, act="relu", RGB=True, has_se=True):
        super().__init__()

        self.keep = [2, 7]
        self.uplayer_shape = [32, 64]
        self.output_channels = 96

        if RGB:
            self.conv1 = nn.Conv2d(
                3, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        else:  # for gray
            self.conv1 = nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        self.bn1 = nn.BatchNorm2d(16)
        if act == "relu":
            self.hs1 = nn.ReLU(inplace=True)
        if has_se:
            self.bneck = nn.Sequential(
                Block3x3(3, 16, 16, 16, self.hs1, None, 2),  # 0 *
                Block3x3(3, 16, 64, 32, self.hs1, None, 2),  # 1
                Block3x3(3, 32, 96, 32, self.hs1, None, 1),  # 2 *
                Block3x3(5, 32, 96, 32, self.hs1, SeModule(32), 2),  # 3
                Block3x3(5, 32, 224, 32, self.hs1, SeModule(32), 1),  # 4
                Block3x3(5, 32, 224, 32, self.hs1, SeModule(32), 1),  # 5
                Block3x3(5, 32, 128, 64, self.hs1, SeModule(64), 1),  # 6
                Block3x3(5, 64, 160, 64, self.hs1, SeModule(64), 1),  # 7 *
                Block3x3(5, 64, 256, 96, self.hs1, SeModule(96), 2),  # 8
            )

        else:
            self.bneck = nn.Sequential(
                Block3x3(3, 16, 16, 16, self.hs1, None, 2),  # 0 *
                Block3x3(3, 16, 72, 24, self.hs1, None, 2),  # 1
                Block3x3(3, 24, 88, 24, self.hs1, None, 1),  # 2 *
                Block3x3(5, 24, 96, 40, self.hs1, None, 2),  # 3
                Block3x3(5, 40, 240, 40, self.hs1, None, 1),  # 4
                Block3x3(5, 40, 240, 40, self.hs1, None, 1),  # 5
                Block3x3(5, 40, 120, 48, self.hs1, None, 1),  # 6
                Block3x3(5, 48, 144, 48, self.hs1, None, 1),  # 7 *
                Block3x3(5, 48, 288, 96, self.hs1, None, 2),  # 8
            )

    def initialize_weights(self):
        print("random init...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        outs = []
        for index, item in enumerate(self.bneck):
            x = item(x)

            if index in self.keep:
                outs.append(x)
        outs.append(x)
        return outs
