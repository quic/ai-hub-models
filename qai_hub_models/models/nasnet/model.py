# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import timm

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "nasnetalarge.tf_in1k"


class NASNet(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = DEFAULT_WEIGHTS):
        model = timm.create_model(checkpoint_path, pretrained=True)
        model.eval()
        return cls(model, transform_input=True)
