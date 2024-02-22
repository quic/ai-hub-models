# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from PIL.Image import Image
from torchvision import transforms

from qai_hub_models.models._shared.imagenet_classifier.model import (
    IMAGENET_DIM,
    ImagenetClassifier,
)
from qai_hub_models.utils.image_processing import normalize_image_transform

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGENET_DIM),
        transforms.ToTensor(),
        normalize_image_transform(),
    ]
)


def preprocess_image(image: Image) -> torch.Tensor:
    """
    Preprocesses images to be run through torch imagenet classifiers
    as prescribed here:
    https://pytorch.org/hub/pytorch_vision_resnet/
    Parameters:
        image: Input image to be run through the classifier model.
    Returns:
        torch tensor to be directly passed to the model.
    """
    out_tensor: torch.Tensor = IMAGENET_TRANSFORM(image)  # type: ignore
    return out_tensor.unsqueeze(0)


class ImagenetClassifierApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with an ImagenetClassifier.

    For a given image input, the app will:
        * Pre-process the image (resize and normalize)
        * Run Imagnet Classification
        * Convert the raw output into probabilities using softmax
    """

    def __init__(self, model: ImagenetClassifier):
        self.model = model

    def predict(self, image: Image) -> torch.Tensor:
        """
        From the provided image or tensor, predict probability distribution
        over the 1k Imagenet classes.

        Parameters:
            image: A PIL Image in RGB format.

        Returns:
            A (1000,) size torch tensor of probabilities, each one corresponding
            to a different Imagenet1K class.
        """

        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = self.model(input_tensor)
        return torch.softmax(output[0], dim=0)
