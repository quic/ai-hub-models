# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class DETR(BaseModel):
    """Exportable DETR model, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_name: str):
        model = DetrForObjectDetection.from_pretrained(ckpt_name)
        return cls(model)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run DETR on `image` and `mask`, and produce high quality detection results.

        Parameters:
            image: Image tensor to run detection on.
            mask: This represents the padding mask. True if padding was applied on that pixel else False.

        Returns:
            logits: [1, 100 (number of predictions), 92 (number of classes)]
            boxes: [1, 100, 4], 4 == (center_x, center_y, w, h)

        """
        predictions = self.model(image, return_dict=False)
        return predictions[0], predictions[1]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 480,
        width: int = 480,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> List[str]:
        return ["logits", "boxes"]
