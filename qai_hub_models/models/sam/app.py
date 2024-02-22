# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Tuple, no_type_check

import numpy as np
import torch

from qai_hub_models.models.sam.model import SAMQAIHMWrapper


class SAMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Segment-Anything Model.

    The app uses 2 models:
        * encoder (Given input image, emmits image embeddings to be used by decoder)
        * decoder (Lightweight decoder, modified to accept and work with fix image size)

    For a given image input, the app will:
        * Prepare: Runs encoder on given image and creates and caches embeddings
        * Generate masks: Uses cached embeddings and generate masks for given points
    """

    @no_type_check
    def __init__(self, model: SAMQAIHMWrapper):
        self.orig_img_size = None
        self.image_embeddings = None
        self.sam_qaihm_wrapper = model
        self.sam_encoder = self.sam_qaihm_wrapper.get_sam_encoder()
        self.sam_decoder = None

    def prepare(self, input_image: np.ndarray, single_mask_mode=True):
        """
        Prepares App for segmentation of given input image
            - Pre-processes input image
            - Initiate Decoder with input image size

        Parameters:
            input_image: np.ndarry
                Input RGB image loaded as numpy array.
            single_mask_mode: bool
                Set decoder to return single mask for given points.
        """
        if self.sam_encoder is None:
            self.sam_encoder = self.sam_qaihm_wrapper.get_sam_encoder()

        preprocessed_image = self.sam_encoder.preprocess_input_image(input_image)
        self.image_embeddings = self.sam_encoder(preprocessed_image)

        # Initialize decoder
        self.orig_img_size = input_image.shape[:2]
        self.sam_decoder = self.sam_qaihm_wrapper.get_sam_decoder(
            self.orig_img_size, single_mask_mode
        )

    def reset(self):
        """Reset app state"""
        self.image_embeddings = None
        self.orig_img_size = None
        self.sam_decoder = None

    def preprocess_point_coordinates(
        self, input_coords: np.ndarray, image_shape: Tuple[int, int]
    ):
        """Peprocesses Point coordinates to work with decoder"""
        if self.sam_encoder is None:
            raise RuntimeError("Encoder is not intialized. Please run `app.prepare`.")
        return torch.Tensor(
            self.sam_encoder.transforms.apply_coords(input_coords, image_shape)
        )

    def predict(self, *args, **kwargs):
        # See generate_mask_from_points.
        return self.generate_mask_from_points(*args, **kwargs)

    def generate_mask_from_points(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate masks from given points

        Parameters:
            point_coords: torch.Tensor of shape [k, 2]
                Point coordinates from input image for segmentation
            point_labels: torch.Tensor of shape [k]
                Point Labels to select/de-select given point for segmentation
                e.g. Corresponding value is 1 if this point is to be included, otherwise 0
        Returns:
            upscaled_masks: torch.Tensor of shape [1, k, <input image spatial dims>]
            score: torch.Tensor of shape [1, k]
            masks: torch.Tensor of shape [1, k, 256, 256]
                Use this low resolution masks to further slice and upscale for resolutions that Decoder is not intialized to.

        Where,
            k = number of points
        """
        if self.sam_decoder is None:
            raise RuntimeError(
                "Please call `prepare_from_image` or `prepare` before calling `segment`."
            )

        # Prepare inputs for decoder
        # Preprocess point co-ordinates for decoder
        point_coords = self.preprocess_point_coordinates(
            np.expand_dims(np.array(point_coords), 0), self.orig_img_size
        )
        point_labels = torch.Tensor(point_labels).unsqueeze(0)
        mask_input = torch.zeros(self.sam_decoder.get_input_spec()["mask_input"][0])
        has_mask_input = torch.zeros((1,))

        upscaled_masks, scores, masks = self.sam_decoder(
            self.image_embeddings,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
        )

        # Reduce noise from generated masks
        upscaled_masks = self.postprocess_mask(upscaled_masks)
        masks = self.postprocess_mask(masks)

        return upscaled_masks, scores, masks

    def postprocess_mask(self, generated_mask: torch.Tensor):
        """Drop masks lower than threshold to minimize noise"""
        return generated_mask > self.sam_qaihm_wrapper.get_sam().mask_threshold
