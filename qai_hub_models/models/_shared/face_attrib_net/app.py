# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


class FaceAttribNetApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FaceAttribNet.

    The app uses 1 model:
        * FaceAttribNet

    For a given image input, the app will:
        * pre-process the image (convert with mean and std dev.)
        * Run FaceAttribNet inference
        * Return output results
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        # See run_inference_on_image.
        return self.run_inference_on_image(*args, **kwargs)

    def run_inference_on_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image.Image,
    ) -> list[npt.NDArray[np.float32]]:
        """
        Return the corresponding output by running inference on input image.

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        Returns:
            If raw_output is true, returns:
                masks: np.ndarray
                    A list of predicted masks.

            Otherwise, returns:
                segmented_images: list[PIL.Image]
                    Images with segmentation map overlaid with an alpha of 0.5.
        """
        assert pixel_values_or_image is not None, "pixel_values_or_image is None"
        img = pixel_values_or_image

        if isinstance(img, Image.Image):
            img_array = np.asarray(img)
        elif isinstance(img, np.ndarray):
            img_array = img
        else:
            raise RuntimeError("Invalid format")

        img_array = img_array.astype("float32") / 255  # image normalization
        img_array = img_array[np.newaxis, ...]
        img_tensor = torch.Tensor(img_array)
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # convert NHWC to NCHW
        pred_res = self.model(img_tensor)

        pred_res_list: list[npt.NDArray[np.float32]] = [
            np.squeeze(out.detach().numpy()) for out in pred_res
        ]
        return pred_res_list
