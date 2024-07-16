# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from qai_hub_models.utils.image_processing import pil_resize_pad, undo_resize_pad


class MidasApp:
    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        input_height: int,
        input_width: int,
    ):
        self.model = model
        self.input_height = input_height
        self.input_width = input_width

    def predict(self, *args, **kwargs):
        return self.estimate_depth(*args, **kwargs)

    def estimate_depth(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> List[Image.Image] | np.ndarray:
        """
        Estimates the depth at each point in an image and produces a heatmap.

        Parameters:
            image: PIL Image to estimate depth.
            raw_output: If set, returns the raw depth estimates instead of a heatmap.

        Returns:
            A heatmap PIL Image or an np array of depth estimates.
            np array will be shape (h, w) where h, w are the dimensions of the input.
            np array will contain raw depth estimates, while PIL image will normalize
            the values and display them as an RGB image.
        """
        resized_image, scale, padding = pil_resize_pad(
            image, (self.input_height, self.input_width)
        )
        image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0)
        prediction = self.model(image_tensor)
        prediction = undo_resize_pad(prediction, image.size, scale, padding)
        numpy_output = prediction.squeeze().cpu().numpy()
        if raw_output:
            return numpy_output
        heatmap = plt.cm.plasma(numpy_output / numpy_output.max())[..., :3]
        return Image.fromarray((heatmap * 255).astype(np.uint8))
