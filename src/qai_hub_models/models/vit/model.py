# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torchvision.models as tv_models
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class VIT(ImagenetClassifier):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__(net=net)

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = tv_models.vit_b_16(weights=weights)
        return cls(net)

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
