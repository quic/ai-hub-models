# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from PIL.Image import Image

from qai_hub_models.models.mobile_vit.model import MobileVIT


class MobileVITApp:
    """Encapsulates the logic for running inference on a MobileVIT model."""

    def __init__(self, model: MobileVIT) -> None:
        self.model = model

    def predict(self, image: Image) -> torch.Tensor:
        feature = self.model.feature_extractor(images=image, return_tensors="pt")
        logits = self.model(feature.pixel_values)
        return torch.softmax(logits[0], dim=0)
