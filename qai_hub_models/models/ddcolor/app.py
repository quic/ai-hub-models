# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.models.ddcolor.model import DDColor


class DDColorApp:
    """
    DDColorApp wraps around a pretrained model to perform image preprocessing and postprocessing.

    Args :
        img(np.ndarray or PIL.Image) : Input RGB image.

    Returns
    -------
        PIL.Image.Image: The final colorized image.

    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        """
        Wrapper method to  call the 'colorize' function.

        Returns :
            PIL.Image.Image: The colorized output image.

        """
        return self.colorize(*args, **kwargs)

    def colorize(self, pil_image: Image.Image) -> Image.Image:
        """
        Perform image colorization using the DDColor model.

        Arg :
            img : Input image in RGB format.

        Returns
        -------
            PIL.Image.Image : colorized output image.
        """
        img = np.array(pil_image)
        height, width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        if pil_image.mode == "L":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            assert pil_image.mode == "RGB", (
                "Only RGB and Grayscale Pillow images are supported"
            )
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # Resize and convert image to grayscale
        height_resized, width_resized = DDColor.get_input_spec()["image"][0][-2:]
        img_resized = cv2.resize(img, (height_resized, width_resized))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate(
            (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
        )
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0)
        )
        output_ab = self.model(tensor_gray_rgb)

        # Resize output and concatenate with original L channel
        output_ab_resized = (
            F.interpolate(output_ab, size=(height, width))[0]
            .float()
            .numpy()
            .transpose(1, 2, 0)
        )
        output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)
        return Image.fromarray(output_img)
