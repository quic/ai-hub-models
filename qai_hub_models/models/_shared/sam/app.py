# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Callable

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models._shared.sam.model_patches import (
    mask_postprocessing as upscale_masks,
)
from qai_hub_models.utils.image_processing import (
    numpy_image_to_torch,
    preprocess_PIL_image,
)


class SAMInputImageLayout(Enum):
    RGB = 0
    BGR = 1


class SAMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Segment-Anything Model.

    The app uses 2 models:
        * encoder (Given input image, emits image embeddings to be used by decoder)
        * decoder (image embeddings --> predicted segmentation masks)
    """

    def __init__(
        self,
        encoder_input_img_size: int,
        mask_threshold: float,
        input_image_channel_layout: SAMInputImageLayout,
        sam_encoder_splits: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        sam_decoder: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        resize_longest_side: Callable,
    ):
        """
        Parameters:
            encoder_input_img_size: int
                The input dimension for images passed the encoder. Height and width are always the same, hence 1 value here.

            mask_threshold:
                Numerical threshold for a pixel in a mask to be considered a positive.

            input_image_channel_layout: SAMInputImageLayout
                Channel layout ("RGB" or "BGR") expected by the encoder.

            sam_encoder_splits:
                SAM encoder split into parts. Must match input & output of each model part generated by qai_hub_models.models.sam.model.SAMEncoderPart

            sam_decoder:
                SAM decoder. Must match input and output of qai_hub_models.models.sam.model.SAMDecoder
                Note that "mask_input" in forward() is not used by this app, so the decoder requires only 3 inputs rather than 4.

            resize_longest_side
                from qai_hub_models.models.sam.model import ResizeLongestSide
                or
                from qai_hub_models.models.mobilesam.model import ResizeLongestSide
        """
        self.sam_encoder_splits = sam_encoder_splits
        self.sam_decoder = sam_decoder
        self.mask_threshold = mask_threshold
        self.encoder_input_img_size = encoder_input_img_size
        self.input_img_size_transform = resize_longest_side(encoder_input_img_size)
        self.input_image_channel_layout = input_image_channel_layout

    def predict(self, *args, **kwargs):
        return self.predict_mask_from_points(*args, **kwargs)

    def predict_mask_from_points(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict segmentation masks from given points and image(s).

        Parameters:
            pixel_values_or_image: torch.Tensor
                PIL image
                or
                numpy array (N H W C x uint8) or (H W C x uint8)
                    channel layout consistent with self.input_image_channel_layout
                or
                pyTorch tensor (N C H W x int8, value range is [0, 255])
                    channel layout consistent with self.input_image_channel_layout

            point_coords: torch.Tensor of shape [k, 2] or [b, k, 2]
                Point coordinates from input image for segmentation

            point_labels: torch.Tensor of shape [k] or [b, k]
                Point Labels to select/de-select given point for segmentation
                e.g. Corresponding value is 1 if this point is to be included, otherwise 0

            return_logits: bool
                If False, returns boolean masks. If true, returns raw fp32 mask predictions.

        Returns:
            upscaled_masks: torch.Tensor of shape [b, k, <input image spatial dims>].
                See parameter return_logits for type info

            scores: torch.Tensor of shape [b, k]
                Mask confidence score

        Where,
            k = number of points
            b = number of input images
        """
        image_embeddings, input_images_original_size = self.predict_embeddings(
            pixel_values_or_image
        )
        return self.predict_mask_from_points_and_embeddings(
            image_embeddings,
            input_images_original_size,
            point_coords,
            point_labels,
            return_logits,
        )

    def predict_embeddings(
        self, pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image]
    ):
        """
        Predict embeddings from given image.

        Parameters:
            pixel_values_or_image: torch.Tensor
                PIL image
                or
                numpy array (N H W C x uint8) or (H W C x uint8)
                    channel layout consistent with self.input_image_channel_layout
                or
                pyTorch tensor (N C H W x int8, value range is [0, 255])
                    channel layout consistent with self.input_image_channel_layout

        Returns:
            image_embeddings: torch.Tensor of shape [b, k, <encoder embed dim>]
                image embeddings

            input_images_original_size: tuple[int, int]
                Original size of input image (BEFORE reshape to fit encoder input size)

        Where,
            k = number of points
            b = number of input images

        Discussion:
            It is faster to run this once on an image (compared to the entire encoder / decoder pipeline)
            if masks will be predicted several times on the same image.
        """
        # Translate input to torch tensor of shape [N, C, H, W]
        if isinstance(pixel_values_or_image, Image):
            pixel_values_or_image = [pixel_values_or_image]
        if isinstance(pixel_values_or_image, list):
            NCHW_int8_torch_frames = torch.cat(
                [
                    preprocess_PIL_image(
                        x.convert(self.input_image_channel_layout.name), False
                    )
                    for x in pixel_values_or_image
                ]
            )
        elif isinstance(pixel_values_or_image, np.ndarray):
            NCHW_int8_torch_frames = numpy_image_to_torch(pixel_values_or_image, False)
        else:
            NCHW_int8_torch_frames = pixel_values_or_image

        # Resize input image to the encoder's desired input size.
        input_images_original_size = (
            NCHW_int8_torch_frames.shape[2],
            NCHW_int8_torch_frames.shape[3],
        )
        input_images = self.input_img_size_transform.apply_image_torch(
            NCHW_int8_torch_frames
        )

        # Normalize input to [0, 1] (must be done after resize)
        input_images = input_images / 255.0

        # Run encoder
        image_embeddings = input_images
        for encoder_part in self.sam_encoder_splits:
            image_embeddings = encoder_part(image_embeddings)

        return image_embeddings, input_images_original_size

    def predict_mask_from_points_and_embeddings(
        self,
        image_embeddings: torch.Tensor,
        input_images_original_size: tuple[int, int],
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        return_logits: bool = False,
    ):
        """
        Predict segmentation masks from given points and image embeddings.

        Parameters:
            image_embeddings: torch.Tensor of shape [b, k, <encoder embed dim>]
                image embeddings

            input_images_original_size: tuple[int, int]
                Original size of input image (BEFORE reshape to fit encoder input size)

            point_coords: torch.Tensor of shape [k, 2] or [b, k, 2]
                Point coordinates from input image for segmentation.

            point_labels: torch.Tensor of shape [k] or [b, k]
                Point Labels to select/de-select given point for segmentation
                e.g. Corresponding value is 1 if this point is to be included, otherwise 0

            return_logits: bool
                If False, returns boolean masks. If true, returns raw fp32 mask predictions.

        Returns:
            upscaled_masks: torch.Tensor of shape [b, k, <input image spatial dims>].
                See parameter return_logits for type info

            scores: torch.Tensor of shape [b, k]
                Mask confidence score

        Where,
            k = number of points
            b = number of input images
        """
        # Expand point_coords and point_labels to include a batch dimension, if necessary
        if len(point_coords.shape) == 2:
            point_coords = torch.unsqueeze(point_coords, 0)
        if len(point_labels.shape) == 1:
            point_labels = torch.unsqueeze(point_labels, 0)

        # Change point coordinates to map to the same pixel in the resized image.
        point_coords = self.input_img_size_transform.apply_coords_torch(
            point_coords, input_images_original_size
        )

        # Run decoder
        masks, scores = self.sam_decoder(image_embeddings, point_coords, point_labels)

        # Upscale masks
        upscaled_masks = upscale_masks(
            masks, self.encoder_input_img_size, input_images_original_size
        )

        # Apply mask threshold
        if not return_logits:
            upscaled_masks = upscaled_masks > self.mask_threshold

        return upscaled_masks, scores
