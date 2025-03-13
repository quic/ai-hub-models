# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import PIL.Image
import torch
from PIL.Image import Image
from torchvision import transforms

from qai_hub_models.models.fcn_resnet50.model import NUM_CLASSES
from qai_hub_models.utils.draw import create_color_map


def preprocess_image(image: Image) -> torch.Tensor:
    """
    Preprocesses images to be run through torch FCN segmenter
    as prescribed here:
    https://pytorch.org/hub/pytorch_vision_resnet/

    Parameters:
        image: Input image to be run through the classifier model.

    Returns:
        torch tensor to be directly passed to the model.
    """
    out_tensor: torch.Tensor = transforms.ToTensor()(image)
    return out_tensor.unsqueeze(0)


class FCN_ResNet50App:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with FCN_ResNet50.

    For a given image input, the app will:
        * Pre-process the image (normalize)
        * Run image segmentation
        * Convert the raw output into probabilities using softmax
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, image: Image, raw_output: bool = False) -> Image | np.ndarray:
        """
        From the provided image or tensor, segment the image

        Parameters:
            image: A PIL Image in RGB format.

        Returns:
            If raw_output is true, returns:
                masks: np.ndarray
                    A list of predicted masks.

            Otherwise, returns:
                segmented_images: list[PIL.Image]
                    Images with segmentation map overlaid with an alpha of 0.5.
        """

        input_tensor = preprocess_image(image)
        output = self.model(input_tensor)
        output = output[0]
        predictions = output.cpu().numpy()

        if raw_output:
            return predictions

        color_map = create_color_map(NUM_CLASSES)
        out = PIL.Image.blend(image, PIL.Image.fromarray(color_map[predictions]), 0.5)

        return out
