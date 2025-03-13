# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
import torch


def draw_points(
    frame: np.ndarray,
    points: np.ndarray | torch.Tensor,
    color: tuple[int, int, int] = (0, 0, 0),
    size: Union[int, list[int]] = 10,
):
    """
    Draw the given points on the frame.

    Parameters:
        frame: np.ndarray
            np array (H W C x uint8, BGR)

        points: np.ndarray | torch.Tensor
            array (N, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (N * 2,) where layout is
                x1, y1, x2, y2, ...

        color: tuple[int, int, int]
            Color of drawn points (RGB)

        size: int
            Size of drawn points

    Returns:
        None; modifies frame in place.
    """
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)
    assert isinstance(size, int) or len(size) == len(points)
    cv_keypoints = []
    for i, (x, y) in enumerate(points):
        curr_size = size if isinstance(size, int) else size[i]
        cv_keypoints.append(cv2.KeyPoint(int(x), int(y), curr_size))

    cv2.drawKeypoints(
        frame,
        cv_keypoints,
        outImage=frame,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_connections(
    frame: np.ndarray,
    points: np.ndarray | torch.Tensor,
    connections: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 1,
):
    """
    Draw connecting lines between the given points on the frame.

    Parameters:
        frame:
            np array (H W C x uint8, BGR)

        points:
            array (N, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (N * 2,) where layout is
                x1, y1, x2, y2, ...

        connections:
            List of points that should be connected by a line.
            Format is [(src point index, dst point index), ...]

        color:
            Color of drawn points (RGB)

        size: int
            Size of drawn connection lines

    Returns:
        None; modifies frame in place.
    """
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)
    point_pairs = [
        ((int(points[i][0]), int(points[i][1])), (int(points[j][0]), int(points[j][1])))
        for (i, j) in connections
    ]
    cv2.polylines(
        frame, np.array(point_pairs), isClosed=False, color=color, thickness=size  # type: ignore[call-overload]
    )


def draw_box_from_corners(
    frame: np.ndarray, corners: np.ndarray | torch.Tensor, color=(0, 0, 0), size=3
):
    """
    Draw a box using the 4 points provided as boundaries.

    Parameters:
        frame: np.ndarray
            np array (H W C x uint8, BGR)

        corners: np.ndarray | torch.Tensor
            array (4, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (8) where layout is
                x1, y1, x2, y2

        color: tuple[int, int, int]
            Color of drawn points and connection lines (BGR)

        size: int
            Size of drawn points and connection lines

    Returns:
        None; modifies frame in place.
    """
    draw_points(frame, corners, color, size)
    draw_connections(frame, corners, [(0, 1), (0, 2), (1, 3), (2, 3)], color, size)


def draw_box_from_xywh(
    frame: np.ndarray,
    box: np.ndarray | torch.Tensor,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
):
    """
    Draw a box using the provided data (center / height / width) to compute the box.

    Parameters:
        frame: np.ndarray
            np array (H W C x uint8, BGR)

        box: np.ndarray | torch.Tensor
            array (4), where layout is
                [xcenter, ycenter, h, w]

        color: tuple[int, int, int]
            Color of drawn points and connection lines (RGB)

        size: int
            Size of drawn points and connection lines

    Returns:
        None; modifies frame in place.
    """
    xc, yc, h, w = box
    TL = [xc - w // 2, yc - h // 2]
    BR = [xc + w // 2, yc + h // 2]
    cv2.rectangle(frame, TL, BR, color, size)


def draw_box_from_xyxy(
    frame: np.ndarray,
    top_left: np.ndarray | torch.Tensor | tuple[int, int],
    bottom_right: np.ndarray | torch.Tensor | tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
    text: Optional[str] = None,
):
    """
    Draw a box using the provided top left / bottom right points to compute the box.

    Parameters:
        frame: np.ndarray
            np array (H W C x uint8, BGR)

        box: np.ndarray | torch.Tensor
            array (4), where layout is
                [xc, yc, h, w]

        color: tuple[int, int, int]
            Color of drawn points and connection lines (RGB)

        size: int
            Size of drawn points and connection lines BGR channel layout

        text: None | str
            Overlay text at the top of the box.

    Returns:
        None; modifies frame in place.
    """
    if not isinstance(top_left, tuple):
        top_left = (int(top_left[0].item()), int(top_left[1].item()))
    if not isinstance(bottom_right, tuple):
        bottom_right = (int(bottom_right[0].item()), int(bottom_right[1].item()))
    cv2.rectangle(frame, top_left, bottom_right, color, size)
    if text is not None:
        cv2.putText(
            frame,
            text,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            size,
        )


def create_color_map(num_classes):
    """
    Assign a random color to each class in the dataset to produce a segmentation mask for drawing.

    Inputs:
        num_classes: Number of colors to produce.

    Returns:
        A list of `num_classes` colors in RGB format.
    """
    np.random.seed(42)  # For reproducible results
    color_map = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # Background class, usually black
    return color_map
