# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class SegmentationOutputEvaluator(BaseEvaluator):
    """Evaluator for comparing segmentation output against ground truth."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        # This evaluator supports only 1 output tensor at a time.
        output = output.argmax(1).cpu()
        assert gt.shape == output.shape
        self.confusion_matrix += self._generate_matrix(gt, output)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = torch.nanmean(Acc)
        return Acc

    def Intersection_over_Union(self):
        return torch.diag(self.confusion_matrix) / (
            torch.sum(self.confusion_matrix, axis=1)
            + torch.sum(self.confusion_matrix, axis=0)
            - torch.diag(self.confusion_matrix)
        )

    def Mean_Intersection_over_Union(self):
        return torch.nanmean(self.Intersection_over_Union())

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(
            self.confusion_matrix
        )
        iu = torch.diag(self.confusion_matrix) / (
            torch.sum(self.confusion_matrix, axis=1)
            + torch.sum(self.confusion_matrix, axis=0)
            - torch.diag(self.confusion_matrix)
        )

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].int() + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def get_accuracy_score(self) -> float:
        return self.Mean_Intersection_over_Union()

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} mIOU"
