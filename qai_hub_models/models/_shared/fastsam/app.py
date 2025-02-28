# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics.models.fastsam.utils import bbox_iou
from ultralytics.utils import ops

from qai_hub_models.utils.image_processing import preprocess_PIL_image


class FastSAMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FastSAM.

    The app uses 1 model:
        * FastSAM

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run FastSAM inference
        * post-process the image
        * display the input and output side-by-side
    """

    def __init__(
        self,
        fastsam_model: Callable[[torch.Tensor], torch.Tensor],
        confidence: float = 0.4,
        iou_threshold: float = 0.9,
        retina_masks: bool = True,
        model_image_input_shape: tuple[int, int] = (640, 640),
    ):
        self.model = fastsam_model
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.retina_masks = retina_masks
        self.model_image_input_shape = model_image_input_shape

    def predict(self, *args, **kwargs):
        # See upscale_image.
        return self.segment_image(*args, **kwargs)

    def segment_image(self, image_path: str) -> tuple[list[Results], FastSAMPrompt]:
        """
        Upscale provided images

        Parameters:
            pixel_values_or_image: torch.Tensor
                Input PIL image (before pre-processing) or pyTorch tensor (after image pre-processing).

        Returns:
            images: list[PIL.Image.Image]
                A list of upscaled images (one for each input image).
        """
        original_image = Image.open(image_path)
        resized_image = original_image.resize(
            (self.model_image_input_shape[0], self.model_image_input_shape[1])
        )
        img = preprocess_PIL_image(resized_image)
        original_image_arr = np.array(original_image)
        raw_boxes, raw_masks = self.model(img)
        nms_out = ops.non_max_suppression(
            raw_boxes,
            self.confidence,
            self.iou_threshold,
            agnostic=False,
            max_det=100,
            nc=1,  # set to 1 class since SAM has no class predictions
            classes=None,
        )

        full_box = torch.zeros(nms_out[0].shape[1], device=nms_out[0].device)
        full_box[2], full_box[3], full_box[4], full_box[6:] = (
            img.shape[3],
            img.shape[2],
            1.0,
            1.0,
        )
        full_box = full_box.view(1, -1)
        critical_iou_index = bbox_iou(
            full_box[0][:4], nms_out[0][:, :4], iou_thres=0.9, image_shape=img.shape[2:]
        )
        if (
            isinstance(critical_iou_index, torch.Tensor)
            and critical_iou_index.numel() != 0
        ):
            full_box[0][4] = nms_out[0][critical_iou_index][:, 4]
            full_box[0][6:] = nms_out[0][critical_iou_index][:, 6:]
            nms_out[0][critical_iou_index] = full_box

        results: list[Results] = []
        for i, pred in enumerate(nms_out):
            orig_img = original_image_arr
            img_path = image_path[i]
            # No predictions, no masks
            if not len(pred):
                masks = None
            elif self.retina_masks:
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], orig_img.shape
                )

                masks = ops.process_mask_native(
                    raw_masks[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2]
                )  # HWC
            else:
                masks = ops.process_mask(
                    raw_masks[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True
                )  # HWC
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], orig_img.shape
                )
            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names="fastsam",
                    boxes=pred[:, :6],
                    masks=masks,
                )
            )
        prompt_process = FastSAMPrompt(image_path, results, device="cpu")
        segmented_result = prompt_process.everything_prompt()
        return segmented_result, prompt_process
