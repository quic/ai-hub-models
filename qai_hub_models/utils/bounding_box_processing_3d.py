# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from numba import njit, prange
from pyquaternion.quaternion import Quaternion


def compute_box_3d(
    dim: np.ndarray, location: np.ndarray, rotation_y: float
) -> np.ndarray:
    """
    Computes the 3D corner coordinates of a bounding box given its dimensions,
    location, and yaw rotation.

    Args:
        dim:
            A NumPy array of shape (3,) representing (height, width, length) of the box.
        location:
            A NumPy array of shape (3,) representing the (x, y, z) centroid
            location of the box in camera coordinates.
        rotation_y:
            A float representing the yaw rotation (around the y-axis) of the box
            in radians.

    Returns:
        np.ndarray: A NumPy array of shape (8, 3) representing the 8 corners
                    of the 3D bounding box in camera coordinates (x, y, z).
    """
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


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


@njit
def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate signed area of triangle formed by points a, b, and c.

    Args:
        a, b, c (np.ndarray): 2D points, with shape (2,).

    Returns:
        area (float): Signed area of the triangle.
    """
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@njit
def sort_vertex(pts: np.ndarray, num_pts: int) -> np.ndarray:
    """Sort vertices counterclockwise around their centroid.

    Args:
        pts (np.ndarray): Vertices array of shape (N, 2).
        num_pts (int): Number of valid vertices.

    Returns:
        sorted_pts (np.ndarray): Sorted vertices of shape (num_pts, 2).
    """
    center = np.zeros(2, dtype=np.float32)
    for i in range(num_pts):
        center += pts[i]
    center /= num_pts

    angles = np.empty(num_pts, dtype=np.float32)
    for i in range(num_pts):
        vec = pts[i] - center
        angles[i] = math.atan2(vec[1], vec[0])

    idx_sorted = np.argsort(angles)
    sorted_pts = np.empty((num_pts, 2), dtype=np.float32)
    for i in range(num_pts):
        sorted_pts[i] = pts[idx_sorted[i]]
    return sorted_pts


@njit
def line_intersection(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[bool, np.ndarray]:
    """Compute intersection point of segments ab and cd.

    Args:
        a, b, c, d (np.ndarray): Endpoints of line segments with shape (2,)

    Returns:
        if intersection occurs, retruns
            bool: True,
            np.ndarray: intersection point with shape (2).
        otherwise, returns
            bool: False,
            np.ndarray: zero vector with shape (2).
    """
    area_abc = triangle_area(a, b, c)
    area_abd = triangle_area(a, b, d)
    if area_abc * area_abd >= 0:
        return False, np.zeros(2)
    area_cda = triangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd
    if area_cda * area_cdb >= 0:
        return False, np.zeros(2)
    t = area_cda / (area_abd - area_abc)
    return True, a + t * (b - a)


@njit
def point_in_quad(pt: np.ndarray, corners: np.ndarray) -> bool:
    """Check if a point lies within a convex quadrilateral.

    Args:
        pt (np.ndarray): Point of shape (2,)
        corners (np.ndarray): 4 corners of the quadrilateral, of shape (4, 2).

    Returns:
        bool: True if point is inside, Otherwise False.
    """
    ab, ad = corners[1] - corners[0], corners[3] - corners[0]
    ap = pt - corners[0]
    abab, abap = ab @ ab, ab @ ap
    adad, adap = ad @ ad, ad @ ap
    return 0 <= abap <= abab and 0 <= adap <= adad


@njit
def rbbox_to_corners(box: np.ndarray) -> np.ndarray:
    """Convert rotated bbox to 4 corner points.

    Args:
        box (np.ndarray): Bounding box in format [x, y, w, h, theta], of shape (5,).

    Returns:
        np.ndarray: 4 corner points of the rotated box, of shape (4, 2).
    """
    angle = box[4]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    w, h = box[2] / 2, box[3] / 2
    dx = np.array([[-w, -h], [-w, h], [w, h], [w, -h]])
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    return dx @ R.T + box[:2]


@njit
def inter(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate intersection area between two rotated boxes.

    Args:
        box1, box2 (np.ndarray): Rotated bounding box in format [x, y, w, h, theta], of shape (5,).

    Returns:
        float: Intersection area.
    """
    c1, c2 = rbbox_to_corners(box1), rbbox_to_corners(box2)
    int_pts = np.zeros((24, 2), dtype=np.float32)
    num_pts = 0

    for i in range(4):
        if point_in_quad(c1[i], c2):
            int_pts[num_pts] = c1[i]
            num_pts += 1
        if point_in_quad(c2[i], c1):
            int_pts[num_pts] = c2[i]
            num_pts += 1

    for i in range(4):
        a, b = c1[i], c1[(i + 1) % 4]
        for j in range(4):
            c, d = c2[j], c2[(j + 1) % 4]
            has_inter, pt = line_intersection(a, b, c, d)
            if has_inter:
                int_pts[num_pts] = pt
                num_pts += 1

    if num_pts < 3:
        return 0.0

    sorted_pts = sort_vertex(int_pts, num_pts)
    area = 0.0
    for i in range(num_pts - 2):
        area += abs(triangle_area(sorted_pts[0], sorted_pts[i + 1], sorted_pts[i + 2]))
    return area


@njit(parallel=True)
def get_bev_iou_matrix(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Calculate bird's-eye-view (BEV) IoU between sets of rotated boxes.

    Args:
        boxes (np.ndarray): Array of N boxes in format [x, y, w, h, theta], of shape (N, 5).
        query_boxes (np.ndarray): Array of K boxes in format [x, y, w, h, theta], of shape (K, 5).

    Returns:
        np.ndarray: IoU matrix of shape (N, K).
    """
    N, K = boxes.shape[0], query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    for i in prange(N):
        for j in range(K):
            a1 = boxes[i, 2] * boxes[i, 3]
            a2 = query_boxes[j, 2] * query_boxes[j, 3]
            inter_area = inter(boxes[i], query_boxes[j])
            iou[i, j] = inter_area / (a1 + a2 - inter_area)
    return iou
