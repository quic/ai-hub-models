# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Collection

import numpy as np
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata


class MPIIGazeEvaluator(BaseEvaluator):
    """Evaluator for gaze estimation on MPIIGaze dataset."""

    def __init__(self):
        self.reset()

    def add_batch(
        self,
        output: Collection[torch.Tensor],
        target: Collection[torch.Tensor],
    ):
        """
        Args:
            output: Predicted gaze angles [batch, 2] (pitch, yaw in radians)
            target: Ground truth gaze angles [batch, 2] (pitch, yaw in radians)
        """
        if isinstance(output, tuple):
            output = output[-1]
        for pred, gt in zip(output, target):
            pred = pred.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()
            self.predictions.append(pred)
            self.targets.append(gt)

    def reset(self):
        """Reset stored predictions and targets"""
        self.predictions = []
        self.targets = []

    def _compute_metrics(self) -> float:
        """Compute mean angular error"""
        if not self.predictions:
            return 0.0
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        return np.mean(self.angular_error(predictions, targets))

    def angular_error(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate angular error (via cosine similarity)."""
        a = self.pitchyaw_to_vector(a) if a.shape[1] == 2 else a
        b = self.pitchyaw_to_vector(b) if b.shape[1] == 2 else b

        ab = np.sum(np.multiply(a, b), axis=1)
        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)

        a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
        b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

        similarity = np.divide(ab, np.multiply(a_norm, b_norm))
        return np.arccos(similarity) * 180.0 / np.pi

    def pitchyaw_to_vector(self, pitchyaws: np.ndarray) -> np.ndarray:
        """Convert pitch and yaw angles to unit gaze vectors."""
        n = pitchyaws.shape[0]
        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((n, 3))
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
        out[:, 1] = sin[:, 0]
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        return out

    def get_accuracy_score(self) -> float:
        return self._compute_metrics()

    def formatted_accuracy(self) -> str:
        error = self._compute_metrics()
        return f"Mean Angular Error: {error:.3f} degrees"

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Mean Angular Error",
            unit="degrees",
            description="Mean angular error between predicted and ground truth gaze directions.",
        )
