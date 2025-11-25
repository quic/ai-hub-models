# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from timm.models import create_model

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "imagenet"
MODEL_ASSET_VERSION = 1


class Sequencer2D(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls):
        model = create_model(
            "sequencer2d_s",
            pretrained=True,
            num_classes=1000,
            in_chans=3,
            scriptable=True,
        ).eval()
        return cls(model)
