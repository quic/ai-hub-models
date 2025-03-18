# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from qai_hub.client import Device
from torch import nn

from qai_hub_models.utils.asset_loaders import SourceAsRoot, wipe_sys_modules
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "facebook/mask2former-swin-tiny-coco-panoptic"
M2F_SOURCE_REPOSITORY = "https://github.com/huggingface/transformers.git"
M2F_SOURCE_REPO_COMMIT = "75f15f39a0434fe7a61385c4677f2700542a7ba6"
# optimize model to run on QNN
M2F_SOURCE_PATCHES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "patches/optimize.diff"))
]


class Mask2Former(BaseModel):
    """
    Mask2Former segmentation
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt: str = DEFAULT_WEIGHTS) -> Mask2Former:
        with SourceAsRoot(
            M2F_SOURCE_REPOSITORY,
            M2F_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=M2F_SOURCE_PATCHES,
        ) as repo_path:

            net_repo_path = Path(repo_path) / "src"
            sys.path.insert(0, str(net_repo_path))

            import transformers

            wipe_sys_modules(transformers)

            from transformers.models.mask2former.modeling_mask2former import (
                Mask2FormerForUniversalSegmentation,
            )

            net = Mask2FormerForUniversalSegmentation.from_pretrained(ckpt)
        return cls(net)

    def forward(self, image: torch.Tensor):
        """
        Predict panoptic segmentation an input `image`.
        Parameters:
            image: A [1, 3, height, width] image with value range of [0, 1], RGB channel layout.
        Returns:
            Raw logit probabilities of classes as a tensor of shape
                [1, num_classes, num_labels].
            Raw logit probabilities of mask as a tensor of shape
                [1, num_classes, modified_height, modified_width],
                where the modified height and width will be some factor smaller
                than the input image.
        """
        out = self.model(image, return_dict=False)

        return out[0], out[1]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 384,
        width: int = 384,
    ):
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["class_idx", "masks"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True"

        return compile_options

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        options = " --compute_unit cpu"
        return profile_options + options
