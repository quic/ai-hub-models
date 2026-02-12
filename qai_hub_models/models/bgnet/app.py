# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> list[Image.Image] | np.ndarray:
        # See segment_image.
        return self.segment_image(*args, **kwargs)

    def segment_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
    ) -> list[Image.Image] | np.ndarray:
        """
        Return the input image with the segmentation mask overlayed on it.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        raw_output
            See "returns" doc section for details.

        Returns
        -------
        masks_or_images : list[Image.Image] | np.ndarray
            If raw_output is true:
                Numpy array of predicted RGB masks. Shape [N, C, H, W], uint8
            Otherwise:
                List of PIL Images with segmentation map overlaid with an alpha of 0.5.
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        # Run prediction
        pred_masks = self.model(NCHW_fp32_torch_frames)
        pred_mask_img = postprocess_masks(pred_masks, NCHW_fp32_torch_frames.shape[-2:])
        if raw_output:
            return pred_mask_img.numpy()

        # Create color map
        color_map = create_color_map(int(pred_mask_img.max().item()) + 1)
        out: list[Image.Image] = []
        for i, img_tensor in enumerate(NHWC_int_numpy_frames):
            out.append(
                Image.blend(
                    Image.fromarray(img_tensor),
                    Image.fromarray(color_map[pred_mask_img[i][0]]),
                    alpha=0.5,
                )
            )
        return out


def postprocess_masks(
    pred_masks: torch.Tensor, input_size: tuple[int, int] | torch.Size
) -> torch.Tensor:
    """
    Process raw model outputs into segmentation masks by resizing,
    converting logits to probabilities

    Parameters
    ----------
    pred_masks
        Raw outputs [N, C, H, W]
    input_size
        Output resolution (height, width)

    Returns
    -------
    masks : torch.Tensor
        Masks [N, C, H, W], uint8
    """
    # Upsample pred mask to original image size
    # Need to upsample in the probability space, not in class labels

    pred_masks = F.interpolate(
        input=pred_masks,
        size=input_size,
        mode="bilinear",
        align_corners=False,
    )

    pred_masks = pred_masks.sigmoid()
    mask_min = pred_masks.amin((1, 2, 3), keepdim=True)
    mask_max = pred_masks.amax((1, 2, 3), keepdim=True)
    pred_masks = (pred_masks - mask_min) / (mask_max - mask_min + 1e-8)

    # convert segmentation mask to RGB image
    return (pred_masks * 255).to(torch.uint8)
