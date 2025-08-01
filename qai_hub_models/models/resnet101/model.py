# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torchvision.models as tv_models

from qai_hub_models.models._shared.imagenet_classifier.model import (
    ImagenetClassifierWithModelBuilder,
)

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class ResNet101(ImagenetClassifierWithModelBuilder):
    model_builder = tv_models.resnet101
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS
