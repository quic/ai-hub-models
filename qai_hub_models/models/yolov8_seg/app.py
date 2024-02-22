# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from ultralytics.utils.ops import process_mask

from qai_hub_models.utils.bounding_box_processing import batched_nms
from qai_hub_models.utils.draw import create_color_map
from qai_hub_models.utils.image_processing import app_to_net_image_inputs


class YoloV8SegmentationApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with YoloV8 segmentation model.

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run Yolo inference
        * By default,
            - post-processes output using non-maximum-suppression
            - applies predicted mask on input image
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            Tuple[
                List[torch.Tensor],
                List[torch.Tensor],
                List[torch.Tensor],
                List[torch.Tensor],
                torch.Tensor,
            ],
        ],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        input_height: int = 640,
        input_width: int = 640,
    ):
        """
        Initialize a YoloV8SegmentationApp application.

        Parameters:
            model: torch.Tensor
                YoloV8 segmentation model.

                Inputs:
                    Tensor of shape (N H W C x float32) with range [0, 1] and BGR channel layout.

                Outputs:
                    boxes: torch.Tensor
                        Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
                    scores: torch.Tensor
                        Class scores multiplied by confidence: Shape is [batch, num_preds]
                    masks: torch.Tensor
                        Predicted masks: Shape is [batch, num_preds, 32]
                    classes: torch.Tensor
                        Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
                    protos: torch.Tensor
                        Tensor of shape[batch, 32, mask_h, mask_w]
                        Multiply masks and protos to generate output masks.

            nms_score_threshold
                Score threshold for non maximum suppression.

            nms_iou_threshold
                Intersection over Union threshold for non maximum suppression.
        """
        self.model = model
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_height = input_height
        self.input_width = input_width

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is valid model input.
        """
        return all([s % 32 == 0 for s in pixel_values.shape[-2:]])

    def preprocess_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        img_size = (self.input_height, self.input_width)
        return Resize(img_size)(pixel_values)

    def predict(self, *args, **kwargs):
        # See predict_boxes_from_image.
        return self.predict_segmentation_from_image(*args, **kwargs)

    def predict_segmentation_from_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | List[Image],
        raw_output: bool = False,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ] | List[Image.Image]:
        """
        From the provided image or tensor, predict the bounding boxes & classes of objects detected within.

        Parameters:
            pixel_values_or_image: torch.Tensor
                PIL image
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both BGR channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), BGR channel layout

            raw_output: bool
                See "returns" doc section for details.

        Returns:
            If raw_output is false or pixel_values_or_image is not a PIL image, returns:
                pred_boxes: List[torch.Tensor]
                    List of predicted boxes for all the batches.
                    Each pred_box is of shape [num_boxes, 4]
                pred_scores: List[torch.Tensor]
                    List of scores for each predicted box for all the batches.
                    Each pred_score is of shape [num_boxes]
                pred_masks: List[torch.Tensor]
                    List of predicted masks for all the batches.
                    Each pred_mask is of shape [num_boxes, 32]
                pred_classes: List[torch.Tensor]
                    List of predicted class for all the batches.
                    Each pred_class is of shape [num_boxes]

            Otherwise, returns:
                image_with_masks: List[PIL.Image]
                    Input image with predicted masks applied
        """

        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        # Cache input spatial dimension to use for post-processing
        input_h, input_w = NCHW_fp32_torch_frames.shape[2:]
        NCHW_fp32_torch_frames = self.preprocess_input(NCHW_fp32_torch_frames)

        self.check_image_size(NCHW_fp32_torch_frames)

        # Run prediction
        pred_boxes, pred_scores, pred_masks, pred_class_idx, proto = self.model(
            NCHW_fp32_torch_frames
        )

        # Non Maximum Suppression on each batch
        pred_boxes, pred_scores, pred_class_idx, pred_masks = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes,
            pred_scores,
            pred_class_idx,
            pred_masks,
        )

        # Process mask and upsample to input shape
        for batch_idx in range(len(pred_masks)):
            pred_masks[batch_idx] = process_mask(
                proto[batch_idx],
                pred_masks[batch_idx],
                pred_boxes[batch_idx],
                (self.input_height, self.input_width),
                upsample=True,
            ).numpy()

        # Resize masks to match with input image shape
        pred_masks = F.interpolate(
            input=torch.Tensor(pred_masks),
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )

        # Return raw output if requested
        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return (pred_boxes, pred_scores, pred_masks, pred_class_idx)

        # Create color map and convert segmentation mask to RGB image
        pred_mask_img = torch.argmax(pred_masks, 1)

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
