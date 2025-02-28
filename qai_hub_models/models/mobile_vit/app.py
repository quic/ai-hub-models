# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch
from PIL.Image import Image

from qai_hub_models.models.mobile_vit.model import MobileVIT


class MobileVITApp:
    """
    Encapsulates the logic for running inference on a MobileVIT model.
    """

    def __init__(self, model: MobileVIT):
        self.model = model

    def predict(self, image: Image):

        feature = self.model.feature_extractor(images=image, return_tensors="pt")
        logits = self.model(feature.pixel_values)
        probabilities = torch.softmax(logits[0], dim=0)

        return probabilities
