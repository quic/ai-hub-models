# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.ops import nms


def batched_nms(
    iou_threshold: float,
    score_threshold: float,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *gather_additional_args,
) -> tuple[list[torch.Tensor], ...]:
    """
    Non maximum suppression over several batches.

    Inputs:
        iou_threshold: float
            Intersection over union (IoU) threshold

        score_threshold: float
            Score threshold (throw away any boxes with scores under this threshold)

        boxes: torch.Tensor
            Boxes to run NMS on. Shape is [B, N, 4], B == batch, N == num boxes, and 4 == (x1, x2, y1, y2)

        scores: torch.Tensor
            Scores for each box. Shape is [B, N], range is [0:1]

        *gather_additional_args: torch.Tensor, ...
            Additional tensor(s) to be gathered in the same way as boxes and scores.
            In other words, each arg is returned with only the elements for the boxes selected by NMS.
            Should be shape [B, N, ...]

    Outputs:
        boxes_out: list[torch.Tensor]
            Output boxes. This is list of tensors--one tensor per batch.
            Each tensor is shape [S, 4], where S == number of selected boxes, and 4 == (x1, x2, y1, y2)

        boxes_out: list[torch.Tensor]
            Output scores. This is list of tensors--one tensor per batch.
            Each tensor is shape [S], where S == number of selected boxes.

        *args : list[torch.Tensor], ...
            "Gathered" additional arguments, if provided.
    """
    scores_out: list[torch.Tensor] = []
    boxes_out: list[torch.Tensor] = []
    args_out: list[list[torch.Tensor]] = (
        [[] for _ in gather_additional_args] if gather_additional_args else []
    )

    for batch_idx in range(0, boxes.shape[0]):
        # Clip outputs to valid scores
        batch_scores = scores[batch_idx]
        scores_idx = torch.nonzero(scores[batch_idx] >= score_threshold).squeeze(-1)
        batch_scores = batch_scores[scores_idx]
        batch_boxes = boxes[batch_idx, scores_idx]
        batch_args = (
            [arg[batch_idx, scores_idx] for arg in gather_additional_args]
            if gather_additional_args
            else []
        )

        if len(batch_scores > 0):
            nms_indices = nms(batch_boxes[..., :4], batch_scores, iou_threshold)
            batch_boxes = batch_boxes[nms_indices]
            batch_scores = batch_scores[nms_indices]
            batch_args = [arg[nms_indices] for arg in batch_args]

        boxes_out.append(batch_boxes)
        scores_out.append(batch_scores)
        for arg_idx, arg in enumerate(batch_args):
            args_out[arg_idx].append(arg)

    return boxes_out, scores_out, *args_out


def compute_box_corners_with_rotation(
    xc: torch.Tensor,
    yc: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    From the provided information, compute the (x, y) coordinates of the box's corners.

    Inputs:
        xc: torch.Tensor
            Center of box (x). Shape is [ Batch ]
        yc: torch.Tensor
            Center of box (y). Shape is [ Batch ]
        w: torch.Tensor
            Width of box. Shape is [ Batch ]
        h: torch.Tensor
            Height of box. Shape is [ Batch ]
        theta: torch.Tensor
            Rotation of box (in radians). Shape is [ Batch ]

    Outputs:
        corners: torch.Tensor
            Computed corners. Shape is (B x 4 x 2),
            where 2 == (x, y)
    """
    batch_size = xc.shape[0]

    # Construct unit square
    points = torch.tensor([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=torch.float32).repeat(
        batch_size, 1, 1
    )  # Construct Unit Square. Shape [B, 2, 4], where 2 == (X, Y)
    points *= torch.stack((w / 2, h / 2), dim=-1).unsqueeze(
        dim=2
    )  # Scale unit square to appropriate height and width

    # Rotate unit square to new coordinate system
    R = torch.stack(
        (
            torch.stack((torch.cos(theta), -torch.sin(theta)), dim=1),
            torch.stack((torch.sin(theta), torch.cos(theta)), dim=1),
        ),
        dim=1,
    )  # Construct rotation matrix
    points = R @ points  # Apply Rotation

    # Adjust box to center around the original center
    points = points + torch.stack((xc, yc), dim=1).unsqueeze(dim=2)

    return points.transpose(-1, -2)


def compute_box_affine_crop_resize_matrix(
    box_corners: torch.Tensor, output_image_size: tuple[int, int]
) -> list[np.ndarray]:
    """
    Computes the affine transform matrices required to crop, rescale,
    and pad the box described by box_corners to fit into an image of the given size without warping.

    Inputs:
        box_corners: torch.Tensor
            Bounding box corners. These coordinates will be mapped to the output image. Shape is [B, 3, 2],
            where B = batch,
                  3 = (top left point, bottom left point, top right point)
              and 2 = (x, y)

        output_image_size: float
            Size of image to which the box should be resized and cropped.

    Outputs:
        affines: list[np.ndarray]
            Computed affine transform matrices. Shape is (2 x 3)
    """
    # Define coordinates for translated image
    network_input_points = np.array(
        [[0, 0], [0, output_image_size[1] - 1], [output_image_size[0] - 1, 0]],
        dtype=np.float32,
    )

    # Compute affine transformation that will map the square to the point
    affines: list[np.ndarray] = []
    for batch in range(box_corners.shape[0]):
        src = box_corners[batch][..., :3].detach().numpy()
        affines.append(cv2.getAffineTransform(src, network_input_points))
    return affines


def box_xywh_to_xyxy(box_cwh: torch.Tensor) -> torch.Tensor:
    """
    Convert center, W, H to top left / bottom right bounding box values.

    Inputs:
        box_cwh: torch.Tensor
            Bounding box. Shape is [B, 2, 2]
            [[xc, yc], [w, h]] * Batch

    Outputs:
        box_xy: torch.Tensor
            Bounding box. Output format is [[x0, y0], [x1, y1]]
    """
    # Convert Xc, Yc, W, H to min and max bounding box values.
    x_center = box_cwh[..., 0, 0]
    y_center = box_cwh[..., 0, 1]
    w = box_cwh[..., 1, 0]
    h = box_cwh[..., 1, 1]

    out = torch.clone(box_cwh)
    out[..., 0, 0] = x_center - w / 2.0  # x0
    out[..., 0, 1] = y_center - h / 2.0  # y0
    out[..., 1, 0] = x_center + w / 2.0  # x1
    out[..., 1, 1] = y_center + h / 2.0  # y1

    return out


def box_xyxy_to_xywh(
    box_xy: torch.Tensor,
) -> torch.Tensor:
    """
    Converts box coordinates to center / width / height notation.

    Inputs:
        box_xy: torch.Tensor
            Bounding box. Shape is [B, 2, 2],
            where B = batch,
                  2 = (point 1, point 2),
              and 2 = (x, y)

    Outputs:
        box_cwh
            Bounding box. Shape is [B, 2, 2],
            [[xc, yc], [w, h]] * Batch
    """
    x0 = box_xy[..., 0, 0]
    y0 = box_xy[..., 0, 1]
    x1 = box_xy[..., 1, 0]
    y1 = box_xy[..., 1, 1]

    out = torch.clone(box_xy)
    out[..., 1, 0] = x1 - x0  # w
    out[..., 1, 1] = y1 - y0  # h
    out[..., 0, 0] = x0 + out[..., 1, 0] / 2  # xc
    out[..., 0, 1] = y0 + out[..., 1, 1] / 2  # yc

    return out


def box_xywh_to_cs(box_cwh: list, aspect_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bbox to center-scale format while maintaining aspect ratio.
    Inputs:
        box_cwh: List
            Bounding box. Shape is [4,]
            [xc, yc, w, h]
        aspect_ratio: float
            ratio between width and height

    Outputs:
        center: np.ndarray
            center for bbox. Shape is [2,]
        scale: np.ndarray
            scale for bbox. Shape is [2,]
    """
    x, y, w, h = box_cwh
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25
    return center, scale


def apply_directional_box_offset(
    offset: float | int | torch.Tensor,
    vec_start: torch.Tensor,
    vec_end: torch.Tensor,
    xc: torch.Tensor,
    yc: torch.Tensor,
):
    """
    Offset the bounding box defined by [xc, yc] by a pre-determined length.
    The offset will be applied in the direction of the supplied vector.

    Inputs:
        offset: torch.Tensor
            Floating point offset to apply to the bounding box, in absolute values.
        vec_start: torch.Tensor
            Starting point of the vector. Shape is [B, 2], where 2 == (x, y)
        vec_end: torch.Tensor
            Ending point of the vector. Shape is [B, 2], where 2 == (x, y)
        xc: torch.Tensor
            x center of box.
        yc: torch.Tensor
            y center of box

    Outputs:
        No return value; xy and yc are modified in place.
    """
    xlen = vec_end[..., 0] - vec_start[..., 0]
    ylen = vec_end[..., 1] - vec_start[..., 1]
    vec_len = torch.sqrt(torch.float_power(xlen, 2) + torch.float_power(ylen, 2))

    xc += offset * (xlen / vec_len)
    yc += offset * (ylen / vec_len)


def get_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Given two tensors of shape (4,) in xyxy format,
    compute the iou between the two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return inter_area / float(boxA_area + boxB_area - inter_area)
