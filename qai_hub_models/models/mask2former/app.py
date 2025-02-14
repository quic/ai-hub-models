# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers.models.mask2former.image_processing_mask2former import (
    compute_segments,
    remove_low_and_no_objects,
)

from qai_hub_models.utils.draw import create_color_map
from qai_hub_models.utils.image_processing import app_to_net_image_inputs


class Mask2FormerApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference for Segmentation.

    For a given image input, the app will:
        * Run inference
        * Convert the output segmentation mask into a visual representation
        * Overlay the segmentation mask onto the image and return it
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ):
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

            raw_output: bool
                See "returns" doc section for details.

        Returns:
            If raw_output is true, returns:
                masks: torch.tensor
                    A list of predicted masks.

            Otherwise, returns:
                segmented_images: list[PIL.Image]
                    Images with segmentation map overlaid with an alpha of 0.5.
        """

        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        # Run prediction
        pred1, pred2 = self.model(NCHW_fp32_torch_frames)

        pred_mask_img = self.post_process_panoptic_segmentation(pred1, pred2)

        if raw_output:
            return pred_mask_img

        # Create color map and convert segmentation mask to RGB image
        # Overlay the segmentation masks on the image.
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

    def post_process_panoptic_segmentation(
        self,
        class_queries_logits: torch.Tensor,
        masks_queries_logits: torch.Tensor,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse=None,
        target_sizes=None,
    ):
        """
        Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Args:
            class_queries_logits(`torch.Tensor`):
                The class probability has shape of [num_batch, num_classes, 134]
            masks_queries_logits(`torch.Tensor`):
                The masks probability has shape of [num_batch, num_classes, reduced_height, reduced_width]
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
            to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
            to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """

        if label_ids_to_fuse is None:
            # logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = F.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs = (
            masks_queries_logits.sigmoid()
        )  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        pred_scores, pred_labels = F.softmax(class_queries_logits, dim=-1).max(-1)

        # Loop over items in batch size
        results = []

        for i in range(batch_size):
            (
                mask_probs_item,
                pred_scores_item,
                pred_labels_item,
            ) = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = (
                    target_sizes[i]
                    if target_sizes is not None
                    else mask_probs_item.shape[1:]
                )
                segmentation = torch.zeros((height, width)) - 1
                results.append(segmentation.unsqueeze(0))
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, _ = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append(segmentation.unsqueeze(0))
        return torch.cat(results, 0)
