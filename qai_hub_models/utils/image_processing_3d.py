# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np


def project_to_image(pts_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Projects 3D points from camera coordinates to 2D image coordinates using
    a projection matrix.

    Args:
        pts_3d:
            A NumPy array of shape (N, 3) representing N 3D points in camera
            coordinates (x, y, z).
        P:
            A NumPy array of shape (3, 4) representing the camera projection
            matrix.

    Returns:
        np.ndarray: A NumPy array of shape (N, 2) representing the projected
                    2D points (u, v) on the image plane.
    """
    # Convert 3D points to homogeneous coordinates (x, y, z, 1)
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1
    )

    # Project to 2D image plane
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)

    # Normalize by the third coordinate (depth)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def ddd2locrot(
    center: np.ndarray,
    alpha: np.ndarray,
    dim_x: np.ndarray,
    depth: np.ndarray,
    calib: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D detection parameters (center, alpha, dimensions, depth)
    to 3D location and rotation_y in camera coordinates.

    Args:
        center (np.ndarray): 2D center (x, y) of the object in the image. Shape (N, 2).
        alpha (np.ndarray): Observation angle of the object. Shape (N, 1).
        dim_x (np.ndarray): height of the object. Shape (N,).
        depth (np.ndarray): Estimated depth of the object. Shape (N, 1).
        calib (np.ndarray): Camera projection matrix P (3x4).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - locations (np.ndarray): 3D location (x, y, z) of the object's bottom-center. Shape (N, 3).
            - rotation_y (np.ndarray): Rotation around the Y-axis in camera coordinates. Shape (N, 1).
    """
    locations = unproject_2d_to_3d(center, depth, calib)
    # Adjust y-coordinate from center to bottom of the 3D bounding box
    locations[:, 1] += dim_x / 2
    rotation_y = alpha2rot_y(alpha, center[:, 0:1], calib[0, 2], calib[0, 0])
    return locations, rotation_y


def unproject_2d_to_3d(
    pt_2d: np.ndarray, depth: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """
    Unprojects a 2D point with known depth back to 3D camera coordinates.

    Args:
        pt_2d (np.ndarray): 2D point (u, v) in image coordinates. Shape (N, 2).
        depth (np.ndarray): The estimated depth (Z-coordinate) of the point
                            in camera space. Shappe (N, 1).
        P (np.ndarray): Camera projection matrix (P), typically 3x4.

    Returns:
        np.ndarray: The 3D point (X, Y, Z) in camera coordinates. Shape (N, 3).
    """
    z = depth[:, 0] - P[2, 3]
    x = (pt_2d[:, 0] * depth[:, 0] - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:, 1] * depth[:, 0] - P[1, 3] - P[1, 2] * z) / P[1, 1]

    pt_3d = np.array([x, y, z], dtype=np.float32).transpose(1, 0)
    return pt_3d


def alpha2rot_y(alpha: np.ndarray, x: np.ndarray, cx: float, fx: float) -> np.ndarray:
    """
    Converts observation angle `alpha` to rotation_y around the Y-axis.

    Args:
        alpha (np.ndarray): Observation angle of the object, ranging [-pi,pi]. shape (N, 1).
                       This is the angle between the ray from camera origin to object
                       center and the Z-axis in camera coordinates.
        x (np.ndarray): X-coordinate of the object's center in pixel coordinates. shape (N, 1).
        cx (float): X-coordinate of the principal point (camera center) in pixels.
        fx (float): Focal length in X-direction in pixels.

    Returns:
        np.ndarray: Rotation `rotation_y` around the Y-axis in camera coordinates,
                ranging [-pi,pi]. shape (N, 1).
    """
    rot_y = alpha + np.arctan2(x - cx, fx)

    # Normalize rot_y to be within [-pi, pi]
    rot_y[rot_y > np.pi] -= 2 * np.pi
    rot_y[rot_y < -np.pi] += 2 * np.pi
    return rot_y
