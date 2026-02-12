# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import timm
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class dla102x(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None) -> Self:
        model = timm.create_model("dla102x", pretrained=True)
        model.eval()
        return cls(model)
