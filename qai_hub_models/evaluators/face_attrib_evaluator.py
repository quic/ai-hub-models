# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata


class FaceAttribNetEvaluator(BaseEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(self) -> None:
        """FaceAttribNetEvaluator constructor"""
        self.reset()

    def reset(self) -> None:
        """Reset the evaluation result variables."""
        self.TP_count: int = 0
        self.total: int = 0

    def add_batch(
        self,
        output: torch.Tensor,
        gt: torch.Tensor,
    ) -> None:
        """
        Adds a batch of model outputs and corresponding ground truth data.

        Parameters
        ----------
        output
            Probability output from the `face_attrib_net` model with range [0, 1], shape (N, M) where
            N is batch_size and M is number of attributes (5).
            5 attributes below in order are included:

            - openness of the left eye.
            - openness of the right eye.
            - presence of eyeglasses.
            - presence of a face mask.
            - presence of sunglasses.
        gt
            A tensor of shape (N, M) containing ground truth labels from `FaceAttribDataset`.
            Attributes follow the same order as `output`.
            Value meanings:

            - 0: Closed / Absent
            - 1: Opened / Present
            - -1: Not available
        """
        assert output.shape == gt.shape

        pred = (output > 0.5).int()
        valid_mask = gt != -1
        matches = (pred == gt) & valid_mask
        _TP_count = int(matches.sum().item())
        _count = int(valid_mask.sum().item())

        self.TP_count += _TP_count
        self.total += _count

    def get_accuracy_score(self) -> float:
        """
        Calculate and return the accuracy score for model evaluation.

        Accuracy is defined as the ratio of true positives (TP) to the total number of samples.
        If no samples are present (i.e., total is 0), the method returns 0.0.

        Returns
        -------
        accuracy_score : float
            Accuracy score in the range [0, 1].
        """
        return (
            (float(self.TP_count) / float(self.total)) * 100 if self.total > 0 else 0.0
        )

    def formatted_accuracy(self) -> str:
        """
        Return accuracy in formatted string

        Returns
        -------
        formatted_string : str
            formatted string of accuracy report
        """
        return f"Accuracy: {self.get_accuracy_score():.3f}"

    def get_metric_metadata(self) -> MetricMetadata:
        """
        Return accuracy in MetricMetadata

        Returns
        -------
        MetricMetadata
        """
        return MetricMetadata(
            name="Attribute Accuracy",
            unit="%",
            description="Correctness between the predicted detection and the label.",
            range=(0.0, 100.0),
            float_vs_device_threshold=10.0,
        )
