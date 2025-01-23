# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torchvision.models as tv_models

from qai_hub_models.models._shared.deeplab.model import DeepLabV3Model

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "COCO_WITH_VOC_LABELS_V1"
NUM_CLASSES = 21


class FCN_ResNet50(DeepLabV3Model):
    """Exportable FCNresNet50 image segmentation applications, end-to-end."""

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> FCN_ResNet50:
        model = tv_models.segmentation.fcn_resnet50(weights=weights)
        model.aux_classifier = None
        return cls(model)
