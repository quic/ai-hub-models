# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torchvision.models as tv_models

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models.common import Precision

MODEL_ID = "squeezenet1_1"
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class SqueezeNet(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> SqueezeNet:
        net = tv_models.squeezenet1_1(weights=weights)
        return cls(net)

    def get_hub_quantize_options(self, precision: Precision) -> str:
        return "--range_scheme min_max"
