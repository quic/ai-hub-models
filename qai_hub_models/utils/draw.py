# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy
import torch


def draw_points(
    frame: numpy.ndarray,
    points: numpy.ndarray | torch.Tensor,
    color: Tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
):
    """
    Draw the given points on the frame.

    Parameters:
        frame: numpy.ndarray
            numpy array (H W C x uint8, BGR)

        points: numpy.ndarray | torch.Tensor
            array (N, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (N * 2,) where layout is
                x1, y1, x2, y2, ...

        color: Tuple[int, int, int]
            Color of drawn points (RGB)

        size: int
            Size of drawn points

    Returns:
        None; modifies frame in place.
    """
    n2 = len(points.shape) == 2
    for i in range(0, len(points) if n2 else len(points) // 2):
        x, y = points[i] if n2 else (points[i * 2], points[i * 2 + 1])
        cv2.circle(frame, (int(x), int(y)), size, color, thickness=size)


def draw_connections(
    frame: numpy.ndarray,
    points: numpy.ndarray | torch.Tensor,
    connections: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
):
    """
    Draw connecting lines between the given points on the frame.

    Parameters:
        frame: numpy.ndarray
            numpy array (H W C x uint8, BGR)

        points: numpy.ndarray | torch.Tensor
            array (N, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (N * 2,) where layout is
                x1, y1, x2, y2, ...

        connections: List[Tuple[int, int]]
            List of points that should be connected by a line.
            Format is [(src point index, dst point index), ...]

        color: Tuple[int, int, int]
            Color of drawn points (RGB)

        size: int
            Size of drawn connection lines

    Returns:
        None; modifies frame in place.
    """
    n2 = len(points.shape) == 2
    for connection in connections:
        x0, y0 = (
            points[connection[0]]
            if n2
            else (points[connection[0] * 2], points[connection[0] * 2 + 1])
        )
        x1, y1 = (
            points[connection[1]]
            if n2
            else (points[connection[1] * 2], points[connection[1] * 2 + 1])
        )
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(frame, (x0, y0), (x1, y1), color, size)


def draw_box_from_corners(
    frame: numpy.ndarray, corners: numpy.ndarray | torch.Tensor, color=(0, 0, 0), size=3
):
    """
    Draw a box using the 4 points provided as boundaries.

    Parameters:
        frame: numpy.ndarray
            numpy array (H W C x uint8, BGR)

        corners: numpy.ndarray | torch.Tensor
            array (4, 2) where layout is
                [x1, y1] [x2, y2], ...
            or
            array (8) where layout is
                x1, y1, x2, y2

        color: Tuple[int, int, int]
            Color of drawn points and connection lines (BGR)

        size: int
            Size of drawn points and connection lines

    Returns:
        None; modifies frame in place.
    """
    draw_points(frame, corners, color, size)
    draw_connections(frame, corners, [(0, 1), (0, 2), (1, 3), (2, 3)], color, size)


def draw_box_from_xywh(
    frame: numpy.ndarray,
    box: numpy.ndarray | torch.Tensor,
    color: Tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
):
    """
    Draw a box using the provided data (center / height / width) to compute the box.

    Parameters:
        frame: numpy.ndarray
            numpy array (H W C x uint8, BGR)

        box: numpy.ndarray | torch.Tensor
            array (4), where layout is
                [xcenter, ycenter, h, w]

        color: Tuple[int, int, int]
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
    frame: numpy.ndarray,
    top_left: numpy.ndarray | torch.Tensor | Tuple[int, int],
    bottom_right: numpy.ndarray | torch.Tensor | Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
    text: Optional[str] = None,
):
    """
    Draw a box using the provided top left / bottom right points to compute the box.

    Parameters:
        frame: numpy.ndarray
            numpy array (H W C x uint8, BGR)

        box: numpy.ndarray | torch.Tensor
            array (4), where layout is
                [xc, yc, h, w]

        color: Tuple[int, int, int]
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
    numpy.random.seed(42)  # For reproducible results
    color_map = numpy.random.randint(0, 256, size=(num_classes, 3), dtype=numpy.uint8)
    color_map[0] = [0, 0, 0]  # Background class, usually black
    return color_map
