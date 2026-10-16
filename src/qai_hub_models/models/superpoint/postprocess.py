# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Dynamic (variable-length) SuperPoint postprocessing.

The model's forward() only computes static-shape, per-pixel dense tensors
(raw score maps and dense descriptor maps) so it can be traced/compiled (and
quantized) for on-device execution -- NMS, border suppression, and everything
with a data-dependent output size (keypoint extraction, descriptor sampling
at sparse locations, matching) live here and run on the host, shared by
``app.py`` (demo/inference) and ``evaluator.py`` (HPatches metrics).
"""

from __future__ import annotations

import cv2
import torch

from qai_hub_models.models.superpoint.external_repos.superpoint.superpoint_pytorch import (
    batched_nms,
    sample_descriptors,
)


def _select_top_k(
    scores: torch.Tensor, k: int, *tensors: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Keep the ``k`` highest-scoring rows of ``scores`` and every tensor in ``tensors``."""
    if scores.shape[0] <= k:
        return (scores, *tensors)
    top_idx = torch.topk(scores, k).indices
    return (scores[top_idx], *(t[top_idx] for t in tensors))


def suppress_and_threshold(
    scores_map: torch.Tensor,
    nms_radius: int,
    remove_borders: int,
) -> torch.Tensor:
    """
    Apply NMS and border suppression to a single image's raw dense score map.

    Parameters
    ----------
    scores_map
        Raw (post-softmax) detection score map, shape ``(H, W)``.
    nms_radius
        Radius (pixels) for non-maximum suppression.
    remove_borders
        Width (pixels) of the border to zero out.

    Returns
    -------
    scores_map : torch.Tensor
        Shape ``(H, W)``, non-maxima zeroed and border pixels set to ``-1``.
    """
    scores_map = batched_nms(scores_map[None], nms_radius)[0]
    if remove_borders:
        pad = remove_borders
        scores_map = scores_map.clone()
        scores_map[:pad] = -1
        scores_map[:, :pad] = -1
        scores_map[-pad:] = -1
        scores_map[:, -pad:] = -1
    return scores_map


def extract_keypoints(
    scores_map: torch.Tensor,
    descriptors_dense: torch.Tensor,
    detection_threshold: float,
    top_k: int,
    stride: int,
    nms_radius: int,
    remove_borders: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a variable-length set of keypoints from a single image's dense outputs.

    Parameters
    ----------
    scores_map
        Raw (post-softmax, pre-NMS, pre-border-suppression) detection score
        map, shape ``(H, W)``.
    descriptors_dense
        Dense L2-normalized descriptor map, shape ``(C, H / stride, W / stride)``.
    detection_threshold
        Minimum score for a pixel to be considered a keypoint.
    top_k
        Maximum number of keypoints to keep (highest-scoring first).
    stride
        Downsampling factor between ``scores_map`` and ``descriptors_dense``.
    nms_radius
        Radius (pixels) for non-maximum suppression.
    remove_borders
        Width (pixels) of the border to exclude from detection.

    Returns
    -------
    keypoints : torch.Tensor
        Shape ``(N, 2)``, ``(x, y)`` pixel coordinates in ``scores_map``'s frame.
    scores : torch.Tensor
        Shape ``(N,)``.
    descriptors : torch.Tensor
        Shape ``(N, C)``, L2-normalized descriptor sampled at each keypoint.
    """
    scores_map = suppress_and_threshold(scores_map, nms_radius, remove_borders)

    idxs = torch.where(scores_map > detection_threshold)
    keypoints = torch.stack(idxs[-2:], dim=-1).flip(1).float()
    kp_scores = scores_map[idxs]
    kp_scores, keypoints = _select_top_k(kp_scores, top_k, keypoints)

    if keypoints.shape[0] == 0:
        descriptors = keypoints.new_zeros((0, descriptors_dense.shape[0]))
    else:
        descriptors = sample_descriptors(
            keypoints[None], descriptors_dense[None], stride
        )[0].transpose(0, 1)

    return keypoints, kp_scores, descriptors


def match_descriptors(desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
    """
    Mutual-nearest-neighbor matching between two L2-normalized descriptor sets.

    Equivalent to OpenCV's ``BFMatcher(NORM_L2, crossCheck=True)`` since, for
    unit-norm vectors, minimizing L2 distance is the same as maximizing dot
    product.

    Parameters
    ----------
    desc_a
        Shape ``(N_a, D)`` L2-normalized descriptors.
    desc_b
        Shape ``(N_b, D)`` L2-normalized descriptors.

    Returns
    -------
    matches : torch.Tensor
        Shape ``(M, 2)`` of ``(index_into_a, index_into_b)`` pairs.
    """
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return desc_a.new_zeros((0, 2), dtype=torch.long)
    similarity = desc_a @ desc_b.transpose(0, 1)
    nn_a_to_b = similarity.argmax(dim=1)
    nn_b_to_a = similarity.argmax(dim=0)
    mutual = nn_b_to_a[nn_a_to_b] == torch.arange(nn_a_to_b.shape[0])
    idx_a = torch.nonzero(mutual, as_tuple=True)[0]
    idx_b = nn_a_to_b[idx_a]
    return torch.stack([idx_a, idx_b], dim=1)


def warp_points_xy(points_xy: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    """Warp ``(x, y)`` points through a 3x3 homography."""
    ones = points_xy.new_ones(points_xy.shape[0], 1)
    homogeneous = torch.cat([points_xy, ones], dim=1)
    warped = homogeneous @ homography.transpose(0, 1)
    return warped[:, :2] / warped[:, 2:3]


def _in_bounds(points_xy: torch.Tensor, shape_hw: tuple[int, int]) -> torch.Tensor:
    h, w = shape_hw
    x, y = points_xy[:, 0], points_xy[:, 1]
    return (x >= 0) & (x < w) & (y >= 0) & (y < h)


def compute_repeatability_pair(
    kp_a: torch.Tensor,
    s_a: torch.Tensor,
    kp_b: torch.Tensor,
    s_b: torch.Tensor,
    homography: torch.Tensor,
    shape_a_hw: tuple[int, int],
    shape_b_hw: tuple[int, int],
    distance_thresh: float,
    keep_k_points: int,
) -> float | None:
    """
    HPatches repeatability for one image pair.

    Port of the official SuperPoint repo's ``detector_evaluation.compute_repeatability``:
    keypoints of A are warped into B's frame via ``homography`` and kept if they land
    inside B; keypoints of B are kept (in their own frame) if warping them by the
    inverse homography lands them inside A. Repeatability is the fraction of points
    (from both sets, symmetrically) that have a match within ``distance_thresh``
    pixels in the other set.
    """
    if kp_a.shape[0] == 0 or kp_b.shape[0] == 0:
        return None

    warped_a = warp_points_xy(kp_a, homography)
    mask_a = _in_bounds(warped_a, shape_b_hw)
    warped_a, s_a = warped_a[mask_a], s_a[mask_a]

    warped_b_back = warp_points_xy(kp_b, torch.inverse(homography))
    mask_b = _in_bounds(warped_b_back, shape_a_hw)
    kp_b, s_b = kp_b[mask_b], s_b[mask_b]

    s_a, warped_a = _select_top_k(s_a, keep_k_points, warped_a)
    s_b, kp_b = _select_top_k(s_b, keep_k_points, kp_b)

    n1, n2 = warped_a.shape[0], kp_b.shape[0]
    if n1 + n2 == 0:
        return None
    if n1 == 0 or n2 == 0:
        return 0.0

    dist = torch.cdist(warped_a, kp_b)
    count1 = int((dist.min(dim=1).values <= distance_thresh).sum().item())
    count2 = int((dist.min(dim=0).values <= distance_thresh).sum().item())
    return (count1 + count2) / (n1 + n2)


def compute_homography_correctness_pair(
    kp_a: torch.Tensor,
    s_a: torch.Tensor,
    d_a: torch.Tensor,
    kp_b: torch.Tensor,
    s_b: torch.Tensor,
    d_b: torch.Tensor,
    homography: torch.Tensor,
    shape_a_hw: tuple[int, int],
    shape_b_hw: tuple[int, int],
    correctness_thresh: float,
    keep_k_points: int,
) -> float:
    """
    HPatches homography-estimation correctness for one image pair.

    Port of the official SuperPoint repo's ``descriptor_evaluation.compute_homography``:
    restrict both keypoint sets to the shared (mutually-visible) region, mutually-match
    their descriptors, RANSAC-fit a homography from the matches (in each image's own,
    unwarped coordinate frame), and score 1.0 if that fitted homography's image-corner
    reprojection agrees with the ground-truth homography's within ``correctness_thresh``
    pixels (else 0.0). Unlike Repeatability, every pair contributes a
    value -- 0.0 on too few matches or a failed RANSAC fit -- matching upstream's
    unconditional per-pair averaging.
    """
    if kp_a.shape[0] == 0 or kp_b.shape[0] == 0:
        return 0.0

    warped_a = warp_points_xy(kp_a, homography)
    mask_a = _in_bounds(warped_a, shape_b_hw)
    kp_a, s_a, d_a = kp_a[mask_a], s_a[mask_a], d_a[mask_a]

    warped_b_back = warp_points_xy(kp_b, torch.inverse(homography))
    mask_b = _in_bounds(warped_b_back, shape_a_hw)
    kp_b, s_b, d_b = kp_b[mask_b], s_b[mask_b], d_b[mask_b]

    s_a, kp_a, d_a = _select_top_k(s_a, keep_k_points, kp_a, d_a)
    s_b, kp_b, d_b = _select_top_k(s_b, keep_k_points, kp_b, d_b)

    matches = match_descriptors(d_a, d_b)
    if matches.shape[0] < 4:
        return 0.0

    matched_a = kp_a[matches[:, 0]].numpy()
    matched_b = kp_b[matches[:, 1]].numpy()
    estimated_h, _ = cv2.findHomography(matched_a, matched_b, cv2.RANSAC)
    if estimated_h is None:
        return 0.0

    h, w = shape_a_hw
    corners = torch.tensor(
        [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]]
    )
    true_corners = warp_points_xy(corners, homography)
    estimated_corners = warp_points_xy(corners, torch.from_numpy(estimated_h).float())
    mean_dist = (true_corners - estimated_corners).norm(dim=1).mean().item()
    return float(mean_dist <= correctness_thresh)
