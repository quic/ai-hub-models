# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from qai_hub_models.extern.numba import njit, prange


def compute_box_3d(
    dim: np.ndarray, location: np.ndarray, rotation_y: float
) -> np.ndarray:
    """
    Computes the 3D corner coordinates of a bounding box given its dimensions,
    location, and yaw rotation.

    Parameters
    ----------
        dim:
            A NumPy array of shape (3,) representing (height, width, length) of the box.
        location:
            A NumPy array of shape (3,) representing the (x, y, z) centroid
            location of the box in camera coordinates.
        rotation_y:
            A float representing the yaw rotation (around the y-axis) of the box
            in radians.

    Returns
    -------
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

    Returns
    -------
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
    return canvas.astype(np.uint8)


def transform_to_matrix(
    translation: list[float],
    rotation: list[float],
    inv: bool = False,
    flat: bool = False,
) -> np.ndarray:
    """
    Converts translation and rotation to a 4x4 transformation matrix.

    Args:
        translation: list[float]
            Translation vector [x, y, z].
        rotation: list[float]
            Quaternion rotation [w, x, y, z].
        inv: bool
            compute the inverse transformation.
        flat: bool,
            use only yaw from rotation for a 2D transformation.

    Returns
    -------
        np.ndarray
            4x4 transformation matrix with shape [4, 4].
    """
    from pyquaternion import Quaternion

    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(
            scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
        ).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R if not inv else R.T
    mat[:3, -1] = t if not inv else R.T @ -t
    return mat


def rotation_3d_in_axis(
    points: torch.Tensor,
    angles: torch.Tensor | float,
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False,
):
    """Rotate points by angles according to axis.

    Parameters
    ----------
        points (torch.Tensor):
            Points of shape (N, M, 3).
        angles (torch.Tensor):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises
    ------
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns
    -------
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, (float, int)):
        angles = torch.full(points.shape[:1], angles)
    elif len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles.item())

    assert (
        len(points.shape) == 3
        and len(angles.shape) == 1
        and points.shape[0] == angles.shape[0]
    ), f"Incorrect shape of points angles: {points.shape}, {angles.shape}"

    assert points.shape[-1] in [
        2,
        3,
    ], f"Points size should be 2 or 3 instead of {points.shape[-1]}"

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis in {1, -2}:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos]),
                ]
            )
        elif axis in {2, -1}:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones]),
                ]
            )
        elif axis in {0, -3}:
            rot_mat_T = torch.stack(
                [
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos]),
                ]
            )
        else:
            raise ValueError(f"axis should in range [-3, -2, -1, 0, 1, 2], got {axis}")
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
    return points_new


def circle_nms(
    dets: np.ndarray, thresh: float, post_max_size: int = 83
) -> torch.LongTensor:
    """Circular NMS.

    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.

    Parameters
    ----------
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int, optional): Max number of prediction to be kept.
            Defaults to 83.

    Returns
    -------
        torch.LongTensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep: list[int] = []
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
        return torch.LongTensor(keep[:post_max_size])

    return torch.LongTensor(keep)


def compute_iou_bev(box_a, box_b):
    """
    Compute the IoU (Intersection over Union) between two rotated 3D bounding boxes in BEV.

    Each box is represented as [x_center, y_center, width, height, rotation].
    This version accurately mimics the CUDA function `boxesioubevLauncher()`.
    """
    x_a, y_a, w_a, h_a, angle_a = box_a
    x_b, y_b, w_b, h_b, angle_b = box_b

    # Convert rotated boxes to corner points
    def get_corners(x, y, w, h, angle):
        """Compute corner points of the rotated rectangle"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        w_half, h_half = w / 2, h / 2
        return np.array(
            [
                [
                    x - w_half * cos_a + h_half * sin_a,
                    y - w_half * sin_a - h_half * cos_a,
                ],
                [
                    x + w_half * cos_a + h_half * sin_a,
                    y + w_half * sin_a - h_half * cos_a,
                ],
                [
                    x + w_half * cos_a - h_half * sin_a,
                    y + w_half * sin_a + h_half * cos_a,
                ],
                [
                    x - w_half * cos_a - h_half * sin_a,
                    y - w_half * sin_a + h_half * cos_a,
                ],
            ]
        )

    corners_a = get_corners(x_a, y_a, w_a, h_a, angle_a)
    corners_b = get_corners(x_b, y_b, w_b, h_b, angle_b)

    # Compute polygon intersection area using Shapely
    from shapely.geometry import Polygon

    poly_a = Polygon(corners_a)
    poly_b = Polygon(corners_b)

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0

    intersection_area = poly_a.intersection(poly_b).area
    union_area = poly_a.area + poly_b.area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def nms_cpu(boxes, scores, thresh=0.4, pre_maxsize=None, post_max_size=None):
    """
    CPU Implementation of 3D NMS (Non-Maximum Suppression) using Rotated IoU.

    Parameters
    ----------
        boxes (np.ndarray or torch.Tensor): [N, 5] (x_center, y_center, width, height, ry).
        scores (np.ndarray or torch.Tensor): [N] Confidence scores.
        thresh (float): IoU threshold for suppression.
        pre_maxsize (int, optional): Maximum number of boxes before NMS.
        _post_maxsize (int, optional): Maximum number of boxes after NMS.

    Returns
    -------
        np.ndarray: Indices of selected boxes after NMS.
    """
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

    # Sort boxes by scores in descending order
    order = np.argsort(-scores)

    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = np.array([compute_iou_bev(boxes[i], boxes[j]) for j in order[1:]])

        # Keep boxes with IoU below the threshold
        remaining = np.where(ious < thresh)[0]
        order = order[remaining + 1]  # Shift indices due to slicing

    keep = np.array(keep, dtype=np.int64).tolist()

    if post_max_size is not None:
        keep = keep[:post_max_size]

    return keep


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Parameters
    ----------
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns
    -------
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def compute_corners(tensor):
    """Initialize 3D boxes and compute corners

    Parameters
    ----------
        tensor (torch.Tensor): Input tensor of shape [N, box_dim].
        box_dim (int): Dimension of each box (default: 9).
        with_yaw (bool): Whether boxes include yaw (default: True).
        origin (tuple): Relative position of box origin (default: (0.5, 0.5, 0)).

    Returns
    -------
        torch.Tensor: Corners of boxes in shape [N, 8, 3].
    """
    dims = tensor[:, 3:6]  # w, l, h
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    ).to(device=dims.device, dtype=dims.dtype)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    corners = rotate_3d_along_axis(corners, tensor[:, 6], axis=2)
    corners += tensor[:, :3].view(-1, 1, 3)
    return corners


def onnx_atan2(y, x):
    # Create a pi tensor with the same device and data type as y
    pi = torch.tensor(torch.pi, device=y.device, dtype=y.dtype)
    half_pi = pi / 2
    eps = 1e-6

    # Compute the arctangent of y/x
    ans = torch.atan(y / (x + eps))

    # Create boolean tensors representing positive, negative, and zero values of y and x
    y_positive = y > 0
    y_negative = y < 0
    x_negative = x < 0
    x_zero = x == 0

    # Adjust ans based on the positive, negative, and zero values of y and x
    ans += torch.where(
        y_positive & x_negative, pi, torch.zeros_like(ans)
    )  # Quadrants I and II
    ans -= torch.where(
        y_negative & x_negative, pi, torch.zeros_like(ans)
    )  # Quadrants III and IV
    ans = torch.where(y_positive & x_zero, half_pi, ans)  # Positive y-axis
    return torch.where(y_negative & x_zero, -half_pi, ans)  # Negative y-axis


def rotate_3d_along_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Parameters
    ----------
        points (torch.Tensor): Points of shape (N, M, 3).
        angles (torch.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.

    Raises
    ------
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns
    -------
        torch.Tensor: Rotated points in shape (N, M, 3)
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis in {2, -1}:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, -rot_sin, zeros]),
                torch.stack([rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = torch.stack(
            [
                torch.stack([zeros, rot_cos, -rot_sin]),
                torch.stack([zeros, rot_sin, rot_cos]),
                torch.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")

    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


@njit
def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate signed area of triangle formed by points a, b, and c.

    Parameters
    ----------
        a, b, c (np.ndarray): 2D points, with shape (2,).

    Returns
    -------
        area (float): Signed area of the triangle.
    """
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@njit
def sort_vertex(pts: np.ndarray, num_pts: int) -> np.ndarray:
    """Sort vertices counterclockwise around their centroid.

    Parameters
    ----------
        pts (np.ndarray): Vertices array of shape (N, 2).
        num_pts (int): Number of valid vertices.

    Returns
    -------
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

    Parameters
    ----------
        a, b, c, d (np.ndarray): Endpoints of line segments with shape (2,)

    Returns
    -------
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

    Parameters
    ----------
        pt (np.ndarray): Point of shape (2,)
        corners (np.ndarray): 4 corners of the quadrilateral, of shape (4, 2).

    Returns
    -------
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

    Parameters
    ----------
        box (np.ndarray): Bounding box in format [x, y, w, h, theta], of shape (5,).

    Returns
    -------
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

    Parameters
    ----------
        box1, box2 (np.ndarray): Rotated bounding box in format [x, y, w, h, theta], of shape (5,).

    Returns
    -------
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

    Parameters
    ----------
        boxes (np.ndarray): Array of N boxes in format [x, y, w, h, theta], of shape (N, 5).
        query_boxes (np.ndarray): Array of K boxes in format [x, y, w, h, theta], of shape (K, 5).

    Returns
    -------
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
