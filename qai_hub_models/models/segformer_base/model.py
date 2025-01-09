# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from torch import nn
from transformers import SegformerForSemanticSegmentation

from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegformerBase(BaseModel):
    """
    Segformer Base segmentation
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt: str = DEFAULT_WEIGHTS) -> SegformerBase:
        net = SegformerForSemanticSegmentation.from_pretrained(ckpt)
        return cls(net)

    def forward(self, image_tensor):
        """
        Predict semantic segmentation an input `image`.

        Parameters:
            image: A [1, 3, height, width] image.

        Returns:
            Raw logit probabilities as a tensor of shape
            [1, num_classes, modified_height, modified_width],
            where the modified height and width will be some factor smaller
            than the input image.
        """
        return self.model(image_tensor, return_dict=False)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["class_logits"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]
