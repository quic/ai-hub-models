# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata
from qai_hub_models.models.hrnet_face.app import refine_keypoints_from_heatmaps
from qai_hub_models.utils.image_processing import denormalize_coordinates_affine


class HRNetFaceEvaluator(BaseEvaluator):
    """Evaluates HRNet-Face model using Normalized Mean Error (NME) metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.nmes: list[float] = []

    def add_batch(self, output: tuple[torch.Tensor], gt: list[Any]) -> None:
        """Processes a batch of model outputs and ground truth.

        Parameters
        ----------
            output (tuple[torch.Tensor]): Model output containing heatmaps shape: [B,K,H,W].
            gt (list[Any]): Ground truth data [centers, scales, landmarks].
                - centers: [B, 2] (x,y centers).
                - scales: [B, 2] (width, height scales).
                - landmarks: [B, 29, 2] (x,y keypoints)
        """
        heatmaps = output[0] if isinstance(output, tuple) else output
        heatmaps_np = heatmaps.detach().cpu().numpy()  # [B, K, H, W]
        centers, scales, landmarks = gt

        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.numpy()

        # Decode keypoints in heatmap coords and map to image coords
        coords_hm = refine_keypoints_from_heatmaps(heatmaps_np)
        B, _, heatmap_w, heatmap_h = heatmaps_np.shape
        preds = np.empty_like(coords_hm)
        for i in range(B):
            preds[i] = denormalize_coordinates_affine(
                coords_hm[i], centers[i], scales[i], 0, (heatmap_w, heatmap_h)
            )

        # Compute and store NME
        nmes = self._compute_nme(preds, landmarks)
        self.nmes.extend(nmes.tolist())

    def _compute_nme(self, preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
        """Computes Normalized Mean Error (NME) for 29 keypoints.

        Parameters
        ----------
            preds (np.ndarray): Predicted keypoints, shape [N, 29, 2].
            gts (np.ndarray): Ground truth keypoints, shape [N, 29, 2].

        Returns
        -------
            np.ndarray: NME values for each sample, normalized by inter-ocular distance.

        Source:
            Adapted from HRNet-Facial-Landmark-Detection:
            https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/master/lib/core/evaluation.py#L36
        """
        N, L, _ = preds.shape
        nmes = np.zeros(N, dtype=np.float32)
        for i in range(N):
            interocular = np.linalg.norm(gts[i, 8] - gts[i, 9])  # Distance between eyes
            err = np.linalg.norm(preds[i] - gts[i], axis=1).sum() / (interocular * L)
            nmes[i] = err
        return nmes

    def get_accuracy_score(self) -> float:
        """Returns average NME score"""
        return float(np.mean(self.nmes))

    def formatted_accuracy(self) -> str:
        return f"Mean NME: {self.get_accuracy_score():.4f}"

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Normalized Mean Error",
            unit="NME",
            description="Average keypoint error normalized by inter-ocular distance (COFW dataset).",
        )
