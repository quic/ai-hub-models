# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for tracking accuracy of a Classifier Model."""

    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes
        self.reset()

    def add_batch(self, output: torch.Tensor, gt: int | torch.Tensor):
        # This evaluator supports only 1 output tensor at a time.
        assert len(output.shape) == 2 and output.shape[-1] == self.num_classes
        gt_tensor = torch.Tensor(gt).unsqueeze(1)
        assert len(gt_tensor.shape) == 2 and gt_tensor.shape[0] == output.shape[0]
        batch_size = output.shape[0]
        self.total_samples += batch_size

        top5 = torch.topk(output, 5).indices
        self.top5_count += torch.sum(top5 == gt_tensor).item()
        self.top1_count += torch.sum(top5[:, :1] == gt_tensor).item()

    def reset(self):
        self.top1_count = 0
        self.top5_count = 0
        self.total_samples = 0

    def top1(self) -> float:
        if self.total_samples == 0:
            return 0
        return self.top1_count / self.total_samples

    def top5(self) -> float:
        if self.total_samples == 0:
            return 0
        return self.top5_count / self.total_samples

    def get_accuracy_score(self) -> float:
        return self.top1()

    def formatted_accuracy(self) -> str:
        return f"{self.top1() * 100:.1f}% (Top 1), {self.top5() * 100:.1f}% (Top 5)"
