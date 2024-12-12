# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch

from qai_hub_models.models._shared.yolo.utils import (
    box_transform_xywh2xyxy_split_input,
    get_most_likely_score,
    transform_box_layout_xywh2xyxy,
)


def yolo_detect_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    use_quantized_postprocessing: bool = False,
):
    """
    Post processing to break newer ultralytics yolo models (e.g. Yolov8, Yolo11) detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores and classes.

    Parameters:
        boxes: torch.Tensor
            Shape is [batch, 4, num_preds] where 4 == [x_center, y_center, w, h]
        scores: torch.Tensor
            Shape is [batch, num_classes, num_preds]
            Each element represents the probability that a given box is
                an instance of a given class.
        use_quantized_postprocessing: bool
            If post-processing a non-quantized model, need to split the bounding box
                processing into multiple smaller tensors due to NPU limitations.
            If quantized, the entire processing can be done on a single tensor.

    Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            class scores multiplied by confidence: Shape is [batch, num_preds]
        class_idx: torch.tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    boxes = torch.permute(boxes, [0, 2, 1])
    scores = torch.permute(scores, [0, 2, 1])

    # Convert boxes to (x1, y1, x2, y2)
    # Doing transform in fp16 requires special logic to keep on NPU
    if use_quantized_postprocessing:
        boxes = box_transform_xywh2xyxy_split_input(boxes[..., 0:2], boxes[..., 2:4])
    else:
        boxes = transform_box_layout_xywh2xyxy(boxes)

    # Get class ID of most likely score.
    scores, class_idx = torch.max(scores, -1, keepdim=False)

    # Quantized model runtime doesn't like int32 outputs, so cast class idx to uint8.
    # This is a no-op for coco models, but for datasets with >255 classes, this
    # should be float32 for the unquantized model.
    class_dtype = torch.uint8 if use_quantized_postprocessing else torch.float32
    return boxes, scores, class_idx.to(class_dtype)


def yolo_segment_postprocess(detector_output: torch.Tensor, num_classes: int):
    """
    Post processing to break Yolo Segmentation output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores, masks and classes.

    Parameters:
        detector_output: torch.Tensor
            The output of Yolo Detection model
            Shape is [batch, k, num_preds]
                where, k = # of classes + 4
                k is structured as follows [boxes (4) : # of classes]
                and boxes are co-ordinates [x_center, y_center, w, h]
        num_classes: int
            number of classes

    Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            Class scores multiplied by confidence: Shape is [batch, num_preds]
        masks: torch.Tensor
            Predicted masks: Shape is [batch, num_preds, 32]
        class_idx: torch.Tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    detector_output = torch.permute(detector_output, [0, 2, 1])
    masks_dim = 4 + num_classes
    boxes = detector_output[:, :, :4]
    scores = detector_output[:, :, 4:masks_dim]
    masks = detector_output[:, :, masks_dim:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = transform_box_layout_xywh2xyxy(boxes)

    # Get class ID of most likely score.
    scores, class_idx = get_most_likely_score(scores)

    return boxes, scores, masks, class_idx
