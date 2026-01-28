# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.face_attrib_evaluator import FaceAttribNetEvaluator
from qai_hub_models.models.face_attrib_net.layers import (
    Conv2dBlock,
    DownsampleBlock,
    HeadBlock,
    NormalBlock,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "face_attrib_net"
MODEL_ASSET_VERSION = "1"
DEFAULT_WEIGHTS = "detection_only_05302025.pt"

OUT_NAMES = [
    "left_openness",
    "right_openness",
    "glasses",
    "mask",
    "sunglasses",
]


class FaceAttribNet(BaseModel):
    def __init__(
        self,
        chan: int,
        blks_per_layer: list[int],
        eye_openness_enable: bool | str = True,
        eyeglasses_enable: bool = True,
        face_mask_enable: bool = True,
        sunglasses_enable: bool = True,
        group_size: int = 32,
        activ_type: str = "prelu",
    ):
        """
        Initializes the `face_attrib_net` model with configurable output attributes.

        Parameters
        ----------
        chan
            Number of channels.

        blks_per_layer
            Number of blocks in each layer.

        eye_openness_enable
            Controls eye openness output:
            - True or "both": Enables both left and right eye openness outputs.
            - False: Disables eye openness outputs.
            - "left": Enables left eye openness output only.
            - "right": Enables right eye openness output only.

        eyeglasses_enable
            If True, enables output for the presence of eyeglasses.

        face_mask_enable
            If True, enables output for the presence of a face mask.

        sunglasses_enable
            If True, enables output for the presence of sunglasses.

        group_size
            Size of the group.

        activ_type
            Type of activation function to use.
            one of ["prelu", "relu", "sigmoid", "none"]
        """
        super().__init__()

        self.chan = chan
        self.head = HeadBlock(chan)
        self.blks_per_layer = blks_per_layer

        self.main_module = nn.ModuleList()
        for i in range(len(self.blks_per_layer)):
            self.main_module.append(self._make_net(self.chan, self.blks_per_layer[i]))
            self.chan *= 2

        self.left_openness = (eye_openness_enable in {"left", "both"}) or (
            eye_openness_enable is True
        )
        if self.left_openness:
            self.base_chan = chan
            self.left_openness_bran1_gconv = Conv2dBlock(
                self.base_chan * 4,
                self.base_chan * 2,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 4 // group_size,
                norm="none",
                activ="none",
            )
            self.left_openness_bran1_conv = Conv2dBlock(
                self.base_chan * 2,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.left_openness_bran2_gconv = Conv2dBlock(
                self.base_chan * 10,
                self.base_chan * 5,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 10 // group_size,
                norm="none",
                activ="none",
            )
            self.left_openness_bran2_conv = Conv2dBlock(
                self.base_chan * 5,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.left_openness_ave = nn.AvgPool2d(kernel_size=4, stride=1)
            self.left_openness_cls = nn.Linear(self.base_chan * 2, 2)

        self.right_openness = (eye_openness_enable == {"right", "both"}) or (
            eye_openness_enable is True
        )
        if self.right_openness:
            self.base_chan = chan
            self.right_openness_bran1_gconv = Conv2dBlock(
                self.base_chan * 4,
                self.base_chan * 2,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 4 // group_size,
                norm="none",
                activ="none",
            )
            self.right_openness_bran1_conv = Conv2dBlock(
                self.base_chan * 2,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.right_openness_bran2_gconv = Conv2dBlock(
                self.base_chan * 10,
                self.base_chan * 5,
                3,
                padding=1,
                stride=2,
                group=self.base_chan * 10 // group_size,
                norm="none",
                activ="none",
            )
            self.right_openness_bran2_conv = Conv2dBlock(
                self.base_chan * 5,
                self.base_chan * 2,
                1,
                padding=0,
                stride=1,
                group=1,
                norm="bn",
                activ=activ_type,
            )
            self.right_openness_ave = nn.AvgPool2d(kernel_size=4, stride=1)
            self.right_openness_cls = nn.Linear(self.base_chan * 2, 2)

        self.glasses = eyeglasses_enable
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

        self.sunglasses = sunglasses_enable
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

        self.mask = face_mask_enable
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

    def _make_net(self, chan: int, n: int) -> nn.Sequential:
        """
        Helper function to instantiate a custom neural network blocks.

        Parameters
        ----------
        chan
            Number of channels.

        n
            Number of blocks included in the layer.

        Returns
        -------
        network_blocks
            custom neural network blocks
        """
        cnn_x: list[nn.Module] = []
        cnn_x += [DownsampleBlock(chan)]
        for _i in range(n - 1):
            cnn_x += [NormalBlock(2 * chan)]
        return nn.Sequential(*cnn_x)

    def forward_intermediate(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass data through model neural network and get intermediate outputs.

        Parameters
        ----------
        image
            image tensor in range [0, 1], shape (N, C, H, W)

        Returns
        -------
        logit
            Shape(N, M, B), where:
            - N: Batch size
            - M: Number of attributes (5)
            - B: 2 (binary classification logits)
                - Index 0: Absent/Closed
                - Index 1: Present/Open

            5 attributes along dim=1 are:
                - left_eye_out : Logits representing the openness of the left eye.
                - right_eye_out : Logits representing the openness of the right eye.
                - eyeglasses_out : Logits indicating the presence of eyeglasses.
                - face_mask_out : Logits indicating the presence of a face mask.
                - sunglasses_out : Logits indicating the presence of sunglasses.

            may contain `NaN` entries if the corresponding attribute is disabled.

        prob
            Shape(N, M). Probabilites

        fea4
            Shape(N, 512, 8, 8). Feature maps.

        """
        x = (image - 0.5) / 0.50196078
        fea1 = self.head(x)

        fea2 = self.main_module[0](fea1)
        fea3 = self.main_module[1](fea2)
        fea4 = self.main_module[2](fea3)

        batch_size = image.shape[0]

        if self.left_openness:
            a = self.left_openness_bran1_conv(self.left_openness_bran1_gconv(fea3))
            a = torch.cat((a, fea4), dim=1)
            a = self.left_openness_ave(
                self.left_openness_bran2_conv(self.left_openness_bran2_gconv(a))
            )
            a = a.flatten(start_dim=1)
            left_eye_out = self.left_openness_cls(a)
        else:
            left_eye_out = torch.full((batch_size, 2), float("nan"))

        if self.right_openness:
            a = self.right_openness_bran1_conv(self.right_openness_bran1_gconv(fea3))
            a = torch.cat((a, fea4), dim=1)
            a = self.right_openness_ave(
                self.right_openness_bran2_conv(self.right_openness_bran2_gconv(a))
            )
            a = a.flatten(start_dim=1)
            right_eye_out = self.right_openness_cls(a)
        else:
            right_eye_out = torch.full((batch_size, 2), float("nan"))

        if self.glasses:
            a = self.eyeglasses_bran1_gconv(fea3)
            a = self.eyeglasses_bran1_conv(a)
            a = torch.cat((a, fea4), dim=1)
            a = self.eyeglasses_ave(
                self.eyeglasses_bran2_conv(self.eyeglasses_bran2_gconv(a))
            )
            a = a.flatten(start_dim=1)
            eye_glasses_out = self.eyeglasses_cls(a)
        else:
            eye_glasses_out = torch.full((batch_size, 2), float("nan"))

        if self.mask:
            a = self.mask_bran1_gconv(fea3)
            a = self.mask_bran1_conv(a)
            a = torch.cat((a, fea4), dim=1)
            a = self.mask_ave(self.mask_bran2_conv(self.mask_bran2_gconv(a)))
            a = a.flatten(start_dim=1)
            face_mask_out = self.mask_cls(a)
        else:
            face_mask_out = torch.full((batch_size, 2), float("nan"))

        if self.sunglasses:
            a = self.sunglasses_bran1_conv(self.sunglasses_bran1_gconv(fea3))
            a = torch.cat((a, fea4), dim=1)
            a = self.sunglasses_ave(
                self.sunglasses_bran2_conv(self.sunglasses_bran2_gconv(a))
            )
            a = a.flatten(start_dim=1)
            sunglasses_out = self.sunglasses_cls(a)
        else:
            sunglasses_out = torch.full((batch_size, 2), float("nan"))

        # attr torch.Tensor, shape (N, M, 2)
        # N: batch size
        # M: number of attributes (5)
        # 2: logits absence (index 0) and presence (index 1)
        logit = torch.stack(
            [
                left_eye_out,
                right_eye_out,
                eye_glasses_out,
                face_mask_out,
                sunglasses_out,
            ],
            dim=1,
        )

        prob = torch.nn.functional.softmax(logit, dim=2)[:, :, 1]
        return logit, prob, fea4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Runs the `face_attrib_net` model on input face images to extract various facial attributes.

        Parameters
        ----------
        image
            Input face images with shape (N, C, H, W), where:
            - N: Batch size
            - C: Number of channels (3 for RGB)
            - H: Height (128 pixels)
            - W: Width (128 pixels)
            Pixel values should be in the range [0, 1].

        Returns
        -------
        prob
            Range [0, 1]
            Shape(N, M), where:
            - N: Batch size
            - M: Number of attributes (5)
            Probabilites of 5 attributes in order:
                - openness of the left eye.
                - openness of the right eye.
                - presence of eyeglasses.
                - presence of a face mask.
                - presence of sunglasses.
            may contain `NaN` entries if the corresponding attribute is disabled.
        """
        _, prob, _ = self.forward_intermediate(image)
        return prob

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None) -> Self:
        """
        Load model weights from a checkpoint and return an instance of FaceAttribNet.

        This method creates a FaceAttribNet model with predefined configuration,
        loads weights from a cached asset store (ignoring the provided checkpoint_path),
        moves the model to CPU, sets it to evaluation mode, and returns the instance.

        Parameters
        ----------
        checkpoint_path
            path to a checkpoint file.
            (Note: currently ignored; uses default asset store path.)

        Returns
        -------
        model
            An initialized and pre-trained model ready for inference.
        """
        faceattribnet_model = cls(
            64,
            [2, 6, 3],
            eye_openness_enable=True,
            eyeglasses_enable=True,
            sunglasses_enable=True,
            face_mask_enable=True,
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
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        """
        Return list of output names associated with forward function

        Returns
        -------
        output_names
            each output name corresponds to one in forward function output.
        """
        return ["probability"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        """
        Return name string for "channel-last" input

        Returns
        -------
        channel_last_inputs
            list of name string of "channel-last" input
        """
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        """
        Return evaluator class for evaluating this model.

        Returns
        -------
        evaluator
            evaluator class for evaluating this model.
        """
        return FaceAttribNetEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        """
        Return list of strings with names of all datasets on which
        this model can be evaluated.

        Returns
        -------
        dataset_names
            list of strings with names of all datasets on which `face_attrib_net` model can be evaluated.
        """
        return ["face_attrib_dataset"]

    @staticmethod
    def calibration_dataset_name() -> str:
        """
        Return the name of the dataset to use for calibration when quantizing the model.

        Returns
        -------
        dataset_name
            name of the calibration dataset

        """
        return "face_attrib_dataset"
