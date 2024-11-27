# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import ConditionalDetrForObjectDetection

from qai_hub_models.models._shared.detr.model import DETR

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "microsoft/conditional-detr-resnet-50"
MODEL_ASSET_VERSION = 1


class ConditionalDETRResNet50(DETR):
    """Exportable DETR model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        return cls(ConditionalDetrForObjectDetection.from_pretrained(ckpt_name))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run DETR on `image` and `mask`, and produce high quality detection results.

        Parameters:
            image: Image tensor to run detection on.
            mask: This represents the padding mask. True if padding was applied on that pixel else False.

        Returns:
            logits: [1, 100 (number of predictions), 92 (number of classes)]
            boxes: [1, 100, 4], 4 == (center_x, center_y, w, h)

        """
        predictions = self.model(image)
        return predictions[0], predictions[1]
