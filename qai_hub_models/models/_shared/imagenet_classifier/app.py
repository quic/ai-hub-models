# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from PIL.Image import Image

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.image_processing import (
    IMAGENET_TRANSFORM,
    normalize_image_transform,
)


def preprocess_image(image: Image, normalize: bool = False) -> torch.Tensor:
    """
    Preprocesses images to be run through torch imagenet classifiers
    as prescribed here:
    https://pytorch.org/hub/pytorch_vision_resnet/
    Parameters:
        image: Input image to be run through the classifier model.

        normalize: bool
            Perform normalization to the standard imagenet mean and standard deviation.
    Returns:
        torch tensor to be directly passed to the model.
    """
    out_tensor = IMAGENET_TRANSFORM(image)
    assert isinstance(out_tensor, torch.Tensor)
    if normalize:
        out_tensor = normalize_image_transform()(out_tensor)

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

    def __init__(
        self,
        model: ExecutableModelProtocol,
        normalization_in_network: bool = True,
    ):
        """
        Parameters:
            model: ExecutableModelProtocol
                The imagenet classifier.

            normalization_in_network: bool
                Whether the classifier normalizes the input using the standard imagenet mean and standard deviation.
                If false, the app will preform the normalization in a preprocessing step.
        """
        self.model = model
        self.normalization_in_network = normalization_in_network

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
        input_tensor = preprocess_image(image, not self.normalization_in_network)
        output = self.model(input_tensor)
        return torch.softmax(output[0], dim=0)
