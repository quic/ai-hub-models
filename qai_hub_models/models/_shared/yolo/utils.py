# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch


def box_transform_xywh2xyxy_split_input(xy: torch.Tensor, wh: torch.Tensor):
    """
    Convert boxes with (xy, wh) layout to (xyxy)

    Parameters
    ----------
        boxes (torch.Tensor): Input boxes with layout (xywh)

    Returns
    -------
        torch.Tensor: Output box with layout (xyxy)
            i.e. [top_left_x | top_left_y | bot_right_x | bot_right_y]
    """
    cx, cy = xy.unbind(dim=-1)
    wh = wh * 0.5
    w, h = wh.unbind(dim=-1)
    return torch.stack((cx - w, cy - h, cx + w, cy + h), -1)


def transform_box_layout_xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes with (xywh) layout to (xyxy)

    Parameters
    ----------
        boxes (torch.Tensor): Input boxes with layout (xywh)

    Returns
    -------
        torch.Tensor: Output box with layout (xyxy)
            i.e. [top_left_x | top_left_y | bot_right_x | bot_right_y]
    """
    # Convert to (x1, y1, x2, y2)
    cx, cy, w, h = boxes.unbind(dim=-1)
    w = w * 0.5
    h = h * 0.5
    return torch.stack((cx - w, cy - h, cx + w, cy + h), -1)


def detect_postprocess(detector_output: torch.Tensor):
    """
    Post processing to break Yolo(v6,v7) detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, classes, and confidence.

    Parameters
    ----------
        detector_output: torch.Tensor
            The output of Yolo Detection model
            Shape is [batch, num_preds, k]
                where, k = # of classes + 5
                k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
                and box_coordinates are [x_center, y_center, w, h]

    Returns
    -------
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            class scores multiplied by confidence: Shape is [batch, num_preds]
        class_idx: torch.tensor
            Predicted class for each bounding box: Shape [batch, num_preds, 1]
    """
    # Break output into parts
    boxes = detector_output[:, :, :4]
    conf = detector_output[:, :, 4:5]
    scores = detector_output[:, :, 5:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = transform_box_layout_xywh2xyxy(boxes)

    # Combine confidence and scores.
    scores *= conf

    # Get class ID of most likely score.
    scores, class_idx = get_most_likely_score(scores)

    return boxes, scores, class_idx.to(torch.uint8)


def detect_postprocess_split_input(
    xy: torch.Tensor,
    wh: torch.Tensor,
    scores: torch.Tensor,
):
    """Same as `detect_postprocess` with inputs split into separate tensors."""
    boxes = box_transform_xywh2xyxy_split_input(xy, wh)

    # For TF Lite, this translates to slice + reshape rather than unpack.
    # This works around a quantization issue where the unpack is left as fp32.
    # TODO(#16306): Move this to regular slice when the quantization issue is fixed.
    conf, per_class_scores = scores.split_with_sizes([1, scores.shape[-1] - 1], dim=-1)

    # Get class ID of most likely score.
    # (#10357) QNN has a bug where passing a result of Mul into ReduceMax returns all 0s
    class_idx_scores, class_idx = torch.max(per_class_scores + 1e-10, -1, keepdim=False)

    # Combine confidence and scores.
    # Original repo does this before the max operation, but that is more
    # expensive by a factor of NUM_CLASSES and mathematically equivalent to this.
    class_idx_scores *= conf.squeeze(-1)

    # Cast classes to int8 for imsdk compatibility
    return boxes, class_idx_scores, class_idx.to(torch.uint8)


def get_most_likely_score(scores: torch.Tensor):
    """
    Returns most likely score and class id

    Parameters
    ----------
        scores (torch.tensor): final score after post-processing predictions

    Returns
    -------
        scores: torch.Tensor
            class scores reduced to keep max score per prediction
            Shape is [batch, num_preds]
        class_idx: torch.tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # TODO(#8595): QNN crashes when running max on a large tensor
    # Split into chunks of size 5k to keep the model NPU resident
    score_splits = torch.split(scores, 5000, dim=-2)
    max_scores = []
    max_indices = []
    for split in score_splits:
        scores, class_idx = torch.max(split, -1, keepdim=False)
        max_scores.append(scores)
        # class_idx needs to be int to make evaluation code work
        max_indices.append(class_idx.int())
    return torch.cat(max_scores, dim=-1), torch.cat(max_indices, dim=-1)
