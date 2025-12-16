# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata


class MaskedLMEvaluator(BaseEvaluator):
    """
    Evaluator for Masked Language Modeling (MLM) top-1 accuracy.

    Measures how often the model correctly predicts the original token
    that was replaced by [MASK].
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset accumulated correct predictions and total count."""
        self._num_correct: int = 0
        self._num_total: int = 0

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor) -> None:
        """
        Accumulate predictions and ground truth

        Parameters
        ----------
        output
            Model output: predicted token IDs at mask positions.
            Shape: [batch_size, 1]
        gt
            Ground truth token IDs (original tokens before masking).
            Shape: [batch_size, 1]
        """
        if not (isinstance(output, torch.Tensor) and isinstance(gt, torch.Tensor)):
            raise TypeError(
                "MaskedLMEvaluator expects torch.Tensor for both output and gt"
            )

        pred_ids = output.detach().cpu().view(-1)
        gt_ids = gt.detach().cpu().view(-1)

        if pred_ids.shape != gt_ids.shape:
            raise ValueError(
                f"Shape mismatch: predictions {pred_ids.shape}, ground truth {gt_ids.shape}"
            )

        correct = (pred_ids == gt_ids).sum().item()
        total = gt_ids.numel()

        self._num_correct += int(correct)
        self._num_total += int(total)

    def get_accuracy_score(self) -> float:
        if self._num_total == 0:
            return 0.0
        return float(self._num_correct) / float(self._num_total)

    def formatted_accuracy(self) -> str:
        return f"{100.0 * self.get_accuracy_score():.2f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Top-1 Masked LM Accuracy",
            unit="%",
            description="Percentage of masked tokens correctly predicted (highest-probability token).",
            range=(0.0, 100.0),
            float_vs_device_threshold=10.0,
        )
