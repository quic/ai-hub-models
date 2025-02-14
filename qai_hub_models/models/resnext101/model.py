# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torchvision.models as tv_models

from qai_hub_models.models._shared.imagenet_classifier.model import (
    ImagenetClassifierWithModelBuilder,
)

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class ResNeXt101(ImagenetClassifierWithModelBuilder):
    model_builder = tv_models.resnext101_32x8d
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS
