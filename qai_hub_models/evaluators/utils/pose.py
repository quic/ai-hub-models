# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import math

import numpy as np

from qai_hub_models.utils.image_processing import (
    apply_affine_to_coordinates,
    compute_affine_transform,
)

BODY_SIGMAS = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]
FOOT_SIGMAS = [0.068, 0.066, 0.066, 0.092, 0.094, 0.094]
FACE_SIGMAS = [
    0.042,
    0.043,
    0.044,
    0.043,
    0.040,
    0.035,
    0.031,
    0.025,
    0.020,
    0.023,
    0.029,
    0.032,
    0.037,
    0.038,
    0.043,
    0.041,
    0.045,
    0.013,
    0.012,
    0.011,
    0.011,
    0.012,
    0.012,
    0.011,
    0.011,
    0.013,
    0.015,
    0.009,
    0.007,
    0.007,
    0.007,
    0.012,
    0.009,
    0.008,
    0.016,
    0.010,
    0.017,
    0.011,
    0.009,
    0.011,
    0.009,
    0.007,
    0.013,
    0.008,
    0.011,
    0.012,
    0.010,
    0.034,
    0.008,
    0.008,
    0.009,
    0.008,
    0.008,
    0.007,
    0.010,
    0.008,
    0.009,
    0.009,
    0.009,
    0.007,
    0.007,
    0.008,
    0.011,
    0.008,
    0.008,
    0.008,
    0.01,
    0.008,
]
LEFTHAND_SIGMAS = [
    0.029,
    0.022,
    0.035,
    0.037,
    0.047,
    0.026,
    0.025,
    0.024,
    0.035,
    0.018,
    0.024,
    0.022,
    0.026,
    0.017,
    0.021,
    0.021,
    0.032,
    0.02,
    0.019,
    0.022,
    0.031,
]
RIGHTHAND_SIGMAS = [
    0.029,
    0.022,
    0.035,
    0.037,
    0.047,
    0.026,
    0.025,
    0.024,
    0.035,
    0.018,
    0.024,
    0.022,
    0.026,
    0.017,
    0.021,
    0.021,
    0.032,
    0.02,
    0.019,
    0.022,
    0.031,
]


def get_final_preds(
    batch_heatmaps: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get final prediction after post process and transform over several batch.

    Inputs:
        batch_heatmaps: numpy.ndarray
            heatmap from the prediction,
            the shape is [batch_size, num_keypoints, height, width]
        center: numpy.ndarray
            center for each image, the shape is [batch_size, 2]
        scale: numpy.ndarray
            scale for each image, the shape is [batch_size, 2]

    Outputs:
        preds: numpy.ndarray
            final prediction based on keypoints after post process and transform,
            the shape is [batch_size, num_keypoints, 2]
        maxval: numpy.ndarray
            score for each prediction, the shape is [batch_size, num_keypoints, 1]
    """
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-process
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px],
                    ]
                )
                coords[n][p] += np.sign(diff) * 0.25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        trans = compute_affine_transform(
            center[i],
            scale[i],
            rot=0,
            output_size=[heatmap_width, heatmap_height],
            inv=True,
        )
        preds[i] = apply_affine_to_coordinates(coords[i], trans)

    return preds, maxvals


def get_max_preds(batch_heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from score maps over several batch.

    Inputs:
        batch_heatmaps: numpy.ndarray
            [batch_size, num_keypoints, height, width])

    Outputs:
        preds: numpy.ndarray
            final prediction based on keypoints,
            the shape is [batch_size, num_keypoints, 2]
        maxval: numpy.ndarray
            score for each prediction,
            the shape is [batch_size, num_keypoints, 1]
    """
    assert isinstance(
        batch_heatmaps, np.ndarray
    ), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # shape [batch, num joints, image h * w]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # index of max pixel per joint
    idx = np.argmax(heatmaps_reshaped, 2)

    # value of max pixel per joint
    maxvals = np.amax(heatmaps_reshaped, 2)

    # Reshape to prep for tiling
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # Tile indices to make room for (x, y)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # Convert index [..., 0] to x
    preds[:, :, 0] = (preds[:, :, 0]) % width
    # Convert [..., 1] to y
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # Tile mask back to (x, y) plane to ignore negatives
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    # apply mask
    preds *= pred_mask
    return preds, maxvals
