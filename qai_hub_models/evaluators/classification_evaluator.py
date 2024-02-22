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
        gt_tensor = torch.Tensor(gt)
        assert len(gt_tensor.shape) == 1 and gt_tensor.shape[0] == output.shape[0]
        batch_size = output.shape[0]
        self.total_samples += batch_size
        self.num_correct += sum(torch.argmax(output, dim=-1) == gt_tensor)

    def reset(self):
        self.num_correct = 0
        self.total_samples = 0

    def get_accuracy_score(self) -> float:
        if self.total_samples == 0:
            return 0
        return self.num_correct / self.total_samples
