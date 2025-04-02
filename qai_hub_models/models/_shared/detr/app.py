# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from PIL.Image import Image, Resampling

from qai_hub_models.models._shared.detr.coco_label_map import LABEL_MAP
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_xyxy
from qai_hub_models.utils.draw import draw_box_from_xyxy
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
    preprocess_PIL_image,
)


class DETRApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with DETR.

    For a given image input, the app will:
        * Preprocess the image (normalize, resize, etc) and get encoding to pass to the model.
        * Run DETR Inference
        * Convert the raw output into box coordinates and corresponding label and confidence.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        model_image_height: Optional[int] = None,
        model_image_width: Optional[int] = None,
    ):
        self.model = model
        self.model_image_height = model_image_height
        self.model_image_width = model_image_width

    def _process_boxes_scores_classes(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_class_idx: torch.Tensor,
        NHWC_int_numpy_frames: list[npt.NDArray[np.uint8]],
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process outputs for (pred_boxes, pred_scores, pred_class_idx) format.

        Parameters:
            pred_boxes: torch.Tensor
                Predicted bounding boxes.
            pred_scores: torch.Tensor
                Predicted scores.
            pred_class_idx: torch.Tensor
                Predicted class indices.
            NHWC_int_numpy_frames: list[npt.NDArray[np.uint8]]
                Image in NHWC format as a list of numpy arrays.
            threshold: float
                Prediction score threshold.

        Returns:
            scores: torch.Tensor
                Confidence scores for the predicted class.
            labels: torch.Tensor
                Labels (class number) for the predicted class.
            boxes: torch.Tensor
                Bounding boxes for the predicted class.
        """
        mask = pred_scores > threshold
        pred_boxes = torch.cat(
            [pred_boxes[i][mask[i]] for i in range(pred_boxes.shape[0])], dim=0
        )
        pred_class_idx = torch.cat(
            [pred_class_idx[i][mask[i]] for i in range(pred_class_idx.shape[0])], dim=0
        )

        # Add boxes to each batch
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            pred_boxes_batch = pred_boxes
            pred_class_idx_batch = pred_class_idx
            if len(pred_boxes_batch.shape) > 0 and len(pred_class_idx_batch.shape) > 0:
                for i, (box, label) in enumerate(
                    zip(pred_boxes_batch, pred_class_idx_batch)
                ):
                    draw_box_from_xyxy(
                        NHWC_int_numpy_frames[batch_idx],
                        box[0:2].int(),
                        box[2:4].int(),
                        color=(0, 255, 0),
                        size=2,
                        text=f"{LABEL_MAP[int(label.item())]}",
                    )

        return pred_scores[mask], pred_class_idx, pred_boxes

    def _process_logits_boxes(
        self,
        out_logits: torch.Tensor,
        out_bbox: torch.Tensor,
        NHWC_int_numpy_frames: list[npt.NDArray[np.uint8]],
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process outputs for [out_logits, out_bbox] format.

        Parameters:
            out_logits: torch.Tensor
                Output logits.
            out_bbox: torch.Tensor
                Output bounding boxes.
            NHWC_int_numpy_frames: list[npt.NDArray[np.uint8]]
                Image in NHWC format as a list of numpy arrays.
            threshold: float
                Prediction score threshold.

        Returns:
            scores: torch.Tensor
                Confidence scores for the predicted class.
            labels: torch.Tensor
                Labels (class number) for the predicted class.
            boxes: torch.Tensor
                Bounding boxes for the predicted class.
        """
        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, _ = prob[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_xywh_to_xyxy(out_bbox.view(-1, 2, 2)).view(-1, 4)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        target_sizes = torch.tensor(NHWC_int_numpy_frames[0].shape[:2][::-1]).unsqueeze(
            0
        )
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        mask = scores > threshold
        labels = torch.argmax(prob[..., :-1], dim=-1)[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        for p, (xmin, ymin, xmax, ymax), l in zip(
            scores.tolist(), boxes.tolist(), labels.tolist()
        ):
            draw_box_from_xyxy(
                NHWC_int_numpy_frames[0],
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color=(0, 255, 0),
                size=2,
                text=f"{LABEL_MAP[l]}: {p:0.2f}",
            )

        return scores, labels, boxes

    def predict(
        self,
        image: Image,
        default_weights: str,
        threshold: float = 0.9,
    ) -> tuple[list[npt.NDArray[np.uint8]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From the provided image or tensor, generate the segmented mask.

        Parameters:
            image: Tensor[B, 3, H, W]
                A PIL Image in NCHW, RGB format.
            default_weights: str
                Default weights name for the model.
            threshold: float
                Prediction score threshold.

        Returns:
            numpy_array: Original image numpy array with the corresponding predictions.
            label: Labels (class number) for the predicted class.
                Shape is [Number of predictions above threshold]
            scores: Confidence scores for the predicted class.
                Shape is [Number of predictions above threshold]
            boxes: Bounding boxes for the predicted class.
                Shape is [Number of predictions above threshold, 4]
        """
        # The official detr demo uses resize instead of padding. There is an option
        # to do padding instead and pass a pixel mask to the model, but we opted for
        # the simpler route.
        # https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=ZRluxbQYYTEe
        if self.model_image_height is not None and self.model_image_width is not None:
            image = image.resize(
                (self.model_image_width, self.model_image_height),
                resample=Resampling.BILINEAR,
            )
        image_array = normalize_image_torchvision(preprocess_PIL_image(image))

        # Convert image to numpy array for drawing
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(image)

        # Run DETR Inference
        outputs = self.model(image_array)

        # Process outputs based on format
        if isinstance(outputs, tuple) and len(outputs) == 3:
            # Process outputs for (pred_boxes, pred_scores, pred_class_idx) format
            scores, labels, boxes = self._process_boxes_scores_classes(
                outputs[0], outputs[1], outputs[2], NHWC_int_numpy_frames, threshold
            )
        else:
            # Process outputs for [out_logits, out_bbox] format
            scores, labels, boxes = self._process_logits_boxes(
                outputs[0], outputs[1], NHWC_int_numpy_frames, threshold
            )

        return NHWC_int_numpy_frames, scores, labels, boxes
