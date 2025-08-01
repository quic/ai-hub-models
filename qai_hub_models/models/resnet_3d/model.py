# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torchvision.models as tv_models

from qai_hub_models.models._shared.video_classifier.model import (
    KineticsClassifier,
    SimpleAvgPool,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS: Any = tv_models.video.R3D_18_Weights.DEFAULT


class ResNet3D(KineticsClassifier):
    @classmethod
    def from_pretrained(
        cls,
        weights: Any = DEFAULT_WEIGHTS,
    ) -> ResNet3D:
        net = tv_models.video.r3d_18(weights=weights)
        net.avgpool = SimpleAvgPool()
        return cls(net)
