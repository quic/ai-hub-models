# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Callable

import torch
from PIL.Image import Image

from qai_hub_models.utils.image_processing import preprocess_PIL_image


class UNetSegmentationApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with UNet.

    For a given image input, the app will:
        * Pre-process the image (resize and normalize)
        * Run UNet Inference
        * Convert the raw output into segmented image.
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, image: Image) -> torch.Tensor:
        """
        From the provided image or tensor, generate the segmented mask.

        Parameters:
            image: A PIL Image in RGB format.

        Returns:
            mask: Segmented mask as numpy array.
        """

        img = preprocess_PIL_image(image)
        out = self.model(img)
        mask = out.argmax(dim=1)
        return mask[0].bool().numpy()
