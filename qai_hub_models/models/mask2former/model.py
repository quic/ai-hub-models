# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
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
from torch.nn import functional as F

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.panoptic_segmentation_evaluator import (
    PanopticSegmentationEvaluator,
)
from qai_hub_models.utils.asset_loaders import SourceAsRoot, wipe_sys_modules
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.image_processing import normalize_image_torchvision

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "facebook/mask2former-swin-tiny-coco-panoptic"
M2F_SOURCE_REPOSITORY = "https://github.com/huggingface/transformers.git"
M2F_SOURCE_REPO_COMMIT = "5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
# optimize model to run on QNN
M2F_SOURCE_PATCHES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "patches/optimize.diff"))
]
NUM_CLASSES = 134


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
        outputs = self.model(normalize_image_torchvision(image), return_dict=False)
        class_logits, mask_logits = outputs[0], outputs[1]
        (class_pred_scores, class_pred_labels) = F.softmax(class_logits, dim=-1).max(-1)
        return class_pred_scores, class_pred_labels, mask_logits

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
        return ["scores", "labels", "masks"]

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
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    def get_evaluator(self) -> BaseEvaluator:
        return PanopticSegmentationEvaluator(NUM_CLASSES)

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["coco_panoptic_seg"]
