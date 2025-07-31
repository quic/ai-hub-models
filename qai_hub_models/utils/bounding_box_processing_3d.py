# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import cv2
import numpy as np
import torch
from pyquaternion.quaternion import Quaternion


def draw_3d_bbox(
    image: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    classes: dict,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw 3D bounding boxes projected onto 2D image from a camera view.

    Agrs:
        image : np.ndarray
            numpy array of original image (H W C x uint8) in RGB channel layout
        coords : np.ndarray
            coordinates of corners in 3D bounding boxes with shape (N, 8, 2)
        labels : np.ndarray
            labels of 3D bounding boxes with shape (N,)
        classes (dict): object classes.
            class_name as key and color as value in tuple (R, G, B) in range[0-255]
        thinkness (int): default is 1.

    returns:
        image : np.ndarray
            numpy array of image with bboxes (H W C x uint8) in RGB channel layout
    """
    canvas = image.copy()

    # draw the bboxes
    for index in range(coords.shape[0]):
        name = list(classes.keys())[labels[index]]

        # edges of bboxes
        for start, end in [
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 5),
            (3, 2),
            (3, 7),
            (4, 5),
            (4, 7),
            (2, 6),
            (5, 6),
            (6, 7),
        ]:
            cv2.line(
                canvas,
                coords[index, start].astype(int),
                coords[index, end].astype(int),
                classes[name],
                thickness,
                cv2.LINE_AA,
            )
    canvas = canvas.astype(np.uint8)
    return canvas


def transform_to_matrix(translation: list, rotation: list) -> np.ndarray:
    """
    Converts translation and rotation to matrix

    Parameters:
        translation: [x, y, z]
        rotation: [w, x, y, z]

    returns:
        matrix: np.ndarray with shape [4, 4]
    """
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, -1] = translation
    return mat


def rotation_3d_in_axis(
    points: torch.Tensor,
    angles: torch.Tensor,
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False,
):
    """Rotate points by angles according to axis.

    Args:
        points (torch.Tensor):
            Points of shape (N, M, 3).
        angles (torch.Tensor):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert (
        len(points.shape) == 3
        and len(angles.shape) == 1
        and points.shape[0] == angles.shape[0]
    ), (f"Incorrect shape of points " f"angles: {points.shape}, {angles.shape}")

    assert points.shape[-1] in [
        2,
        3,
    ], f"Points size should be 2 or 3 instead of {points.shape[-1]}"

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos]),
                ]
            )
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones]),
                ]
            )
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack(
                [
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos]),
                ]
            )
        else:
            raise ValueError(
                f"axis should in range " f"[-3, -2, -1, 0, 1, 2], got {axis}"
            )
    else:
        rot_mat_T = torch.stack(
            [torch.stack([rot_cos, rot_sin]), torch.stack([-rot_sin, rot_cos])]
        )

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum("aij,jka->aik", points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum("jka->ajk", rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


def circle_nms(
    dets: torch.Tensor, thresh: float, post_max_size: int = 83
) -> torch.Tensor:
    """Circular NMS.

    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.

    Args:
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int, optional): Max number of prediction to be kept.
            Defaults to 83.

    Returns:
        torch.Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return keep[:post_max_size]

    return keep
