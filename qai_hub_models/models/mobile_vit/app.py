# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch
from PIL.Image import Image


class MobileVITApp:
    """
    Encapsulates the logic for running inference on a MobileVIT model.

    """

    def __init__(self, model):
        self.model = model

    def predict(self, image: Image):

        image = self.model.feature_extractor(images=image, return_tensors="pt")
        logits = self.model(image.pixel_values)
        probabilities = torch.softmax(logits[0], dim=0)

        return probabilities
