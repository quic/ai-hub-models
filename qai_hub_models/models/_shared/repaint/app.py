# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    torch_tensor_to_PIL_image,
)


class RepaintMaskApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with AOTGAN.

    The app uses 1 model:
        * AOTGAN

    For a given image input, the app will:
        * pre-process the image
        * Run AOTGAN inference
        * Convert the output tensor into a PIL Image
    """

    def __init__(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        # See paint_mask_on_image.
        return self.paint_mask_on_image(*args, **kwargs)

    @staticmethod
    def preprocess_inputs(
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | List[Image],
        mask_pixel_values_or_image: torch.Tensor | np.ndarray | Image,
    ) -> Dict[str, torch.Tensor]:
        NCHW_fp32_torch_frames = app_to_net_image_inputs(pixel_values_or_image)[1]
        NCHW_fp32_torch_masks = app_to_net_image_inputs(mask_pixel_values_or_image)[1]

        # The number of input images should equal the number of input masks.
        if NCHW_fp32_torch_masks.shape[0] != 1:
            NCHW_fp32_torch_masks = NCHW_fp32_torch_masks.tile(
                (NCHW_fp32_torch_frames.shape[0], 1, 1, 1)
            )

        # Mask input image
        image_masked = (
            NCHW_fp32_torch_frames * (1 - NCHW_fp32_torch_masks) + NCHW_fp32_torch_masks
        )
        return {"image": image_masked, "mask": NCHW_fp32_torch_masks}

    def paint_mask_on_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | List[Image],
        mask_pixel_values_or_image: torch.Tensor | np.ndarray | Image,
    ) -> List[Image]:
        """
        Erases and repaints the source image[s] in the pixel values given by the mask.

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

            mask_pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

                If one mask is provided, it will be used for every input image.

        Returns:
            images: List[PIL.Image]
                A list of predicted images (one list element per batch).
        """
        inputs = self.preprocess_inputs(
            pixel_values_or_image, mask_pixel_values_or_image
        )
        out = self.model(inputs["image"], inputs["mask"])

        return [torch_tensor_to_PIL_image(img) for img in out]
