# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torchvision.models as tv_models

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class WideResNet50(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> WideResNet50:
        net = tv_models.wide_resnet50_2(weights=weights)
        return cls(net)
