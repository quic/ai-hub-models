# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.utils.draw import create_color_map
from qai_hub_models.utils.image_processing import app_to_net_image_inputs


class BGNetApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference for Segmentation.

    The app uses model:
        * BGNet


    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run inference
        * Convert the output segmentation mask into a visual representation
        * Overlay the segmentation mask onto the image and return it
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        # See segment_image.
        return self.segment_image(*args, **kwargs)

    def segment_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: bool = False,
    ) -> list[Image.Image] | np.ndarray:
        """
        Return the input image with the segmentation mask overlayed on it.

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        Returns:

                segmented_images: list[PIL.Image]
                    Images with segmentation map overlaid with an alpha of 0.5.
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        # Run prediction
        pred_masks = self.model(NCHW_fp32_torch_frames)
        if isinstance(pred_masks, tuple):
            pred_masks = pred_masks[0]

        pred_mask_img = postprocess_masks(pred_masks, NCHW_fp32_torch_frames.shape[-2:])
        # Create color map
        color_map = create_color_map(pred_mask_img.max().item() + 1)
        out = []
        for i, img_tensor in enumerate(NHWC_int_numpy_frames):
            out.append(
                Image.blend(
                    Image.fromarray(img_tensor),
                    Image.fromarray(color_map[pred_mask_img[i]]),
                    alpha=0.5,
                )
            )
        return out


def postprocess_masks(
    pred_masks: torch.Tensor, input_size: tuple[int, int]
) -> torch.Tensor:
    """
    Process raw model outputs into segmentation masks by resizing,
    converting logits to probabilities

    Args:
        pred_masks: Raw outputs [N, C, H, W]
        input_size: Output resolution (height, width)

    Returns:
        torch.Tensor: Masks [N, H, W], uint8
    """
    # Upsample pred mask to original image size
    # Need to upsample in the probability space, not in class labels

    pred_masks = F.interpolate(
        input=pred_masks,
        size=input_size,
        mode="bilinear",
        align_corners=False,
    )

    pred_masks = pred_masks.sigmoid().squeeze(0)
    pred_masks = (pred_masks - pred_masks.min()) / (
        pred_masks.max() - pred_masks.min() + 1e-8
    )
    # convert segmentation mask to RGB image
    pred_mask_img = (pred_masks * 255).to(torch.uint8)

    return pred_mask_img
