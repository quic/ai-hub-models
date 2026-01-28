# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import BeitForImageClassification
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "microsoft/beit-base-patch16-224"
MODEL_ASSET_VERSION = 1


class Beit(ImagenetClassifier):
    """Exportable Beit model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        return cls(BeitForImageClassification.from_pretrained(ckpt_name))

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(image_tensor, return_dict=False)[0]
