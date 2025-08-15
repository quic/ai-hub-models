# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torchvision.models as tv_models

from qai_hub_models.models._shared.deeplab.model import DeepLabV3Model

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "COCO_WITH_VOC_LABELS_V1"


class DeepLabV3_ResNet50(DeepLabV3Model):
    """Exportable DeepLabV3_ResNet50 image segmentation applications, end-to-end."""

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> DeepLabV3_ResNet50:
        model = tv_models.segmentation.deeplabv3_resnet50(weights=weights)
        return cls(model)
