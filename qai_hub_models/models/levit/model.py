# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import LevitForImageClassification

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "facebook/levit-128S"


class LeViT(ImagenetClassifier):
    """Exportable LeViT model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> LeViT:
        model = LevitForImageClassification.from_pretrained(ckpt_name)
        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for an input `image`.

        Parameters:
            image: A [1, 3, 224, 224] image.
                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1] if self.normalize_input, else ~[-2.5, 2.5]
                   3-channel Color Space: RGB

        Returns:
            A [1, 1000] where each value is the log-likelihood of
            the image belonging to the corresponding Imagenet class.
        """
        predictions = self.net(image, return_dict=False)
        return predictions[0]
