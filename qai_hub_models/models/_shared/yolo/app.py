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
from torchvision.transforms import Resize
from ultralytics.utils.ops import process_mask

from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.utils.bounding_box_processing import batched_nms
from qai_hub_models.utils.draw import create_color_map, draw_box_from_xyxy
from qai_hub_models.utils.image_processing import app_to_net_image_inputs


class YoloObjectDetectionApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with Yolo object detection models.

    The app works with following models:
        * YoloV7
        * YoloV8Detection

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run Yolo inference
        * if requested, post-process YoloV7 output using non maximum suppression
        * if requested, draw the predicted bounding boxes on the input image
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        model_includes_postprocessing: bool = True,
    ):
        """
        Initialize a YoloObjectDetectionApp application.

        Parameters:
            model:
                Yolo object detection model.

                Inputs:
                    Tensor of shape (N H W C x float32) with range [0, 1] and BGR channel layout.

                Outputs:
                    boxes: Tensor of shape [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
                                The output are in the range of the input image's dimensions (NOT [0-1])

                    scores: Tensor of shape [batch, num_preds, # of classes (typically 80)]

                    class_idx: Tensor of shape [num_preds] where the values are the indices
                                of the most probable class of the prediction.

            nms_score_threshold
                Score threshold for non maximum suppression.

            nms_iou_threshold
                Intersection over Union threshold for non maximum suppression.

            model_includes_postprocessing
                Whether the model includes postprocessing steps beyond the detector.
        """
        self.model = model
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.model_includes_postprocessing = model_includes_postprocessing

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is valid model input.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        # See predict_boxes_from_image.
        return self.predict_boxes_from_image(*args, **kwargs)

    def predict_boxes_from_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[
        np.ndarray
    ]:
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
                images: list[np.ndarray]
                    A list of predicted BGR, [H, W, C] images (one list element per batch). Each image will have bounding boxes drawn.

            Otherwise, returns:
                boxes: list[torch.Tensor]
                    Bounding box locations per batch. List element shape is [num preds, 4] where 4 == (x1, y1, x2, y2)
                scores: list[torch.Tensor]
                    class scores per batch multiplied by confidence: List element shape is [num_preds, # of classes (typically 80)]
                class_idx: list[torch.tensor]
                    Shape is [num_preds] where the values are the indices of the most probable class of the prediction.
        """

        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        self.check_image_size(NCHW_fp32_torch_frames)

        # Run prediction
        if self.model_includes_postprocessing:
            pred_boxes, pred_scores, pred_class_idx = self.model(NCHW_fp32_torch_frames)
        else:
            model_output: tuple[torch.Tensor, ...] = self.model(NCHW_fp32_torch_frames)
            if isinstance(model_output, torch.Tensor):
                model_output = (model_output,)
            pred_boxes, pred_scores, pred_class_idx = self.pre_nms_postprocess(
                *model_output
            )

        # Non Maximum Suppression on each batch
        pred_boxes, pred_scores, pred_class_idx = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes,
            pred_scores,
            pred_class_idx,
        )

        # Return raw output if requested
        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            print(pred_boxes, pred_scores, pred_class_idx)
            return (pred_boxes, pred_scores, pred_class_idx)

        # Add boxes to each batch
        for batch_idx in range(len(pred_boxes)):
            pred_boxes_batch = pred_boxes[batch_idx]
            for box in pred_boxes_batch:
                draw_box_from_xyxy(
                    NHWC_int_numpy_frames[batch_idx],
                    box[0:2].int(),
                    box[2:4].int(),
                    color=(0, 255, 0),
                    size=2,
                )

        return NHWC_int_numpy_frames

    def pre_nms_postprocess(
        self, *predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the output of the YOLO detector for input to NMS.

        Parameters:
            predictions: torch.Tensor
                A tuple of tensor outputs from the Yolo detection model.
                Tensor shapes vary by model implementation.

        Returns:
            boxes: torch.Tensor
                Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
            scores: torch.Tensor
                class scores multiplied by confidence: Shape is [batch, num_preds]
            class_idx: torch.Tensor
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
        """
        return detect_postprocess(predictions[0])


class YoloSegmentationApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with Yolo segmentation model.

    The app works with following models:
        * YoloV8Segmentation
        * YoloV11Segmentation

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
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        input_height: int = 640,
        input_width: int = 640,
    ):
        """
        Initialize a YoloSegmentationApp application.

        Parameters:
            model:
                Yolo Segmentation model

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

    def check_image_size(self, pixel_values: torch.Tensor) -> bool:
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
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: bool = False,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ] | list[Image.Image]:
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
                pred_boxes: list[torch.Tensor]
                    List of predicted boxes for all the batches.
                    Each pred_box is of shape [num_boxes, 4]
                pred_scores: list[torch.Tensor]
                    List of scores for each predicted box for all the batches.
                    Each pred_score is of shape [num_boxes]
                pred_masks: list[torch.Tensor]
                    List of predicted masks for all the batches.
                    Each pred_mask is of shape [num_boxes, 32]
                pred_classes: list[torch.Tensor]
                    List of predicted class for all the batches.
                    Each pred_class is of shape [num_boxes]

            Otherwise, returns:
                image_with_masks: list[PIL.Image]
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
