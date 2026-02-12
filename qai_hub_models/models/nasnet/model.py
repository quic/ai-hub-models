# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import timm
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "nasnetalarge.tf_in1k"


class NASNet(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, checkpoint_path: str = DEFAULT_WEIGHTS) -> Self:
        model = timm.create_model(checkpoint_path, pretrained=True)
        model.eval()
        return cls(model, transform_input=True)

    @staticmethod
    def get_hub_litemp_percentage(_: Any) -> float:
        """
        Returns the Lite-MP percentage value for the specified mixed precision quantization.
        The returned value is a constant 5.0
        """
        return 5.0
