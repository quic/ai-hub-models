# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.superpoint.postprocess import (
    compute_homography_correctness_pair,
    compute_repeatability_pair,
    extract_keypoints,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import ACCURACY, MetricMetadata

# This evaluator runs everything at the model's on-device input resolution
# (480x640, see model.DEFAULT_HEIGHT/DEFAULT_WIDTH) rather than reproducing
# upstream's per-metric configs exactly:
#   - Repeatability: same algorithm and NMS/keep_k as upstream's
#     configs/magic-point_repeatability.yaml (nms=4, keep_k_points=300,
#     distance_thresh=3), but upstream evaluates it at 240x320. Since NMS
#     radius and the pixel distance threshold are resolution-dependent, this
#     recipe's Repeatability number is NOT directly comparable to the
#     0.662/0.674 reported in the upstream README -- it reflects the deployed
#     on-device resolution instead of reproducing the paper's config.
#   - Homography Correctness: same algorithm, NMS, and keep_k_points as
#     upstream's configs/superpoint_hpatches.yaml (nms=8, keep_k_points=1000,
#     correctness_thresh=3), evaluated at the same 480x640 resolution
#     upstream uses for this config. It is therefore directly comparable to
#     the 0.965 (illumination) / 0.712 (viewpoint) numbers in the upstream
#     README.
DEFAULT_DETECTION_THRESHOLD = 0.001
DEFAULT_REPEATABILITY_TOP_K = 300
DEFAULT_MATCHING_TOP_K = 1000
DEFAULT_REPEATABILITY_DISTANCE_THRESH = 3.0
DEFAULT_HOMOGRAPHY_CORRECTNESS_THRESH = 3.0
DEFAULT_STRIDE = 8
DEFAULT_REPEATABILITY_NMS_RADIUS = 4
DEFAULT_DESCRIPTOR_NMS_RADIUS = 8
DEFAULT_REMOVE_BORDERS = 4


class HPatchesEvaluator(BaseEvaluator):
    """
    Evaluator for SuperPoint on the HPatches sequences benchmark.

    Computes two metrics per image pair from the model's dense (scores,
    descriptors) outputs for both images, using the dynamic (variable-length)
    keypoint extraction and matching logic in ``postprocess.py``:

    - Repeatability, extracted with a detector-oriented NMS/keep_k config.
    - Homography Correctness, extracted with a descriptor-oriented NMS/keep_k
      config.

    Repeatability is reported as the primary accuracy score
    (``get_accuracy_score`` / ``get_metric_metadata``); Homography
    Correctness is available via ``get_homography_correctness_score`` and
    included in ``formatted_accuracy``.
    """

    def __init__(
        self,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        repeatability_top_k: int = DEFAULT_REPEATABILITY_TOP_K,
        matching_top_k: int = DEFAULT_MATCHING_TOP_K,
        repeatability_distance_thresh: float = DEFAULT_REPEATABILITY_DISTANCE_THRESH,
        homography_correctness_thresh: float = DEFAULT_HOMOGRAPHY_CORRECTNESS_THRESH,
        stride: int = DEFAULT_STRIDE,
        repeatability_nms_radius: int = DEFAULT_REPEATABILITY_NMS_RADIUS,
        descriptor_nms_radius: int = DEFAULT_DESCRIPTOR_NMS_RADIUS,
        remove_borders: int = DEFAULT_REMOVE_BORDERS,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.repeatability_top_k = repeatability_top_k
        self.matching_top_k = matching_top_k
        self.repeatability_distance_thresh = repeatability_distance_thresh
        self.homography_correctness_thresh = homography_correctness_thresh
        self.stride = stride
        self.repeatability_nms_radius = repeatability_nms_radius
        self.descriptor_nms_radius = descriptor_nms_radius
        self.remove_borders = remove_borders
        self.reset()

    def reset(self) -> None:
        self._repeatabilities: list[float] = []
        self._homography_correctness: list[float] = []

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gt: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        output
            ``(scores_a, descriptors_a, scores_b, descriptors_b)`` as returned
            by ``SuperPoint.forward()``.
        gt
            Homographies mapping image_a's pixel coordinates to image_b's,
            shape ``(batch, 3, 3)``, in the same (resized) coordinate frame as
            the model inputs.
        """
        scores_a, descriptors_a, scores_b, descriptors_b = output
        homographies = gt

        for i in range(scores_a.shape[0]):
            homography = homographies[i].cpu().float()
            shape_a = (int(scores_a.shape[-2]), int(scores_a.shape[-1]))
            shape_b = (int(scores_b.shape[-2]), int(scores_b.shape[-1]))

            kp_a_rep, s_a_rep, _ = extract_keypoints(
                scores_a[i].cpu(),
                descriptors_a[i].cpu(),
                self.detection_threshold,
                self.repeatability_top_k,
                self.stride,
                self.repeatability_nms_radius,
                self.remove_borders,
            )
            kp_b_rep, s_b_rep, _ = extract_keypoints(
                scores_b[i].cpu(),
                descriptors_b[i].cpu(),
                self.detection_threshold,
                self.repeatability_top_k,
                self.stride,
                self.repeatability_nms_radius,
                self.remove_borders,
            )

            repeatability = compute_repeatability_pair(
                kp_a_rep,
                s_a_rep,
                kp_b_rep,
                s_b_rep,
                homography,
                shape_a,
                shape_b,
                self.repeatability_distance_thresh,
                self.repeatability_top_k,
            )
            if repeatability is not None:
                self._repeatabilities.append(repeatability)

            kp_a_desc, s_a_desc, d_a_desc = extract_keypoints(
                scores_a[i].cpu(),
                descriptors_a[i].cpu(),
                self.detection_threshold,
                self.matching_top_k,
                self.stride,
                self.descriptor_nms_radius,
                self.remove_borders,
            )
            kp_b_desc, s_b_desc, d_b_desc = extract_keypoints(
                scores_b[i].cpu(),
                descriptors_b[i].cpu(),
                self.detection_threshold,
                self.matching_top_k,
                self.stride,
                self.descriptor_nms_radius,
                self.remove_borders,
            )

            homography_correctness = compute_homography_correctness_pair(
                kp_a_desc,
                s_a_desc,
                d_a_desc,
                kp_b_desc,
                s_b_desc,
                d_b_desc,
                homography,
                shape_a,
                shape_b,
                self.homography_correctness_thresh,
                self.matching_top_k,
            )
            self._homography_correctness.append(homography_correctness)

    def get_repeatability_score(self) -> float:
        """Repeatability (%), averaged over evaluated pairs."""
        if not self._repeatabilities:
            return 0.0
        return float(np.mean(self._repeatabilities)) * 100.0

    def get_accuracy_score(self) -> float:
        """Repeatability (%), averaged over evaluated pairs. Reported as the primary accuracy score."""
        return self.get_repeatability_score()

    def get_homography_correctness_score(self) -> float:
        """Homography-estimation Correctness (%), averaged over ALL evaluated pairs."""
        if not self._homography_correctness:
            return 0.0
        return float(np.mean(self._homography_correctness)) * 100.0

    def formatted_accuracy(self) -> str:
        return (
            f"Repeatability: {self.get_accuracy_score():.3f}%  |  "
            f"Homography Correctness: {self.get_homography_correctness_score():.3f}%  "
            f"(pairs: {len(self._repeatabilities)})"
        )

    def get_metric_metadata(self) -> MetricMetadata:
        return ACCURACY.with_description(
            "HPatches Repeatability: the fraction of keypoints detected in one "
            "image that have a corresponding detected keypoint in the other "
            "image within a pixel distance threshold, after warping by the "
            "ground-truth homography."
        )
