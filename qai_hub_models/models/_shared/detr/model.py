# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Tuple

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
        model.eval()
        return cls(model)

    def forward(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run DETR on `image` and `mask`, and produce high quality detection results.

        Parameters:
            image: Image tensor to run detection on.
            mask: This represents the padding mask. True if padding was applied on that pixel else False.

        Returns:
            predictions: Tuple of tensors (logits and coordinates)
               Shape of logit tensor: [1, 100 (number of predictions), 92 (number of classes)]
               Shape of coordinates: [1, 100, 4]

        """
        predictions = self.model(image, mask, return_dict=False)
        return predictions

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 480,
        width: int = 480,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {
            "image": ((batch_size, num_channels, height, width), "float32"),
            "mask": ((batch_size, height, width), "float32"),
        }
