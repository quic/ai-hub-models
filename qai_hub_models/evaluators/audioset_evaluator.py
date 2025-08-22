# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class AudioSetOutputEvaluator(BaseEvaluator):
    """Evaluator for AudioSet multi-label classification."""

    def __init__(self, num_classes: int = 521):
        """
        Initialize evaluator for AudioSet.

        Args:
            num_classes: Number of AudioSet classes (default: 521).
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.preds = defaultdict(list)
        self.targets = defaultdict(list)

    def add_batch(self, output: torch.Tensor, gt: tuple[torch.Tensor, list[str]]):
        """
        Add a batch of predictions and ground truth for evaluation.

        Args:
            output: Model raw output scores (B, num_classes).
            gt: Ground truth multi-label binary tensor (B, num_classes).
        """
        output = output.cpu()
        label_tensor, sample_ids = gt
        label_tensor = label_tensor.cpu()

        for out, target, sid in zip(output, label_tensor, sample_ids):
            self.preds[sid].append(out)
            if not self.targets[sid]:
                self.targets[sid].append(target)

    def compute_mAP(self) -> float:
        """
        Compute mean Average Precision (mAP) across all accumulated samples.

        Returns:
            float: mAP score
        """
        aggregated_preds = []
        aggregated_targets = []
        for sid in self.preds:
            pred = torch.stack(self.preds[sid]).mean(dim=0)
            target = self.targets[sid][0]
            aggregated_preds.append(pred)
            aggregated_targets.append(target)

        preds = torch.stack(aggregated_preds).numpy()
        targets = torch.stack(aggregated_targets).int().numpy()
        aps = []
        for i in range(self.num_classes):
            if targets[:, i].sum() > 0:  # Skip classes with no positive targets
                ap = average_precision_score(targets[:, i], preds[:, i])
                aps.append(ap)
        return np.mean(aps)

    def get_accuracy_score(self) -> float:
        """Return mAP"""
        return self.compute_mAP()

    def formatted_accuracy(self) -> str:
        """Return formatted mAP score."""
        return f"{self.get_accuracy_score():.3f} mAP"
