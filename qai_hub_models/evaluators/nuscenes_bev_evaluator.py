# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.segmentation_evaluator import SegmentationOutputEvaluator

VEHICLE_INDICES = [4, 5, 6, 7, 8, 10, 11]


class NuscenesBevSegmentationEvaluator(SegmentationOutputEvaluator):
    """Evaluator for Bird's Eye View (BEV) segmentation on the NuScenes dataset."""

    def __init__(
        self,
        iou_thresholds: list[float] | None = None,
        num_classes: int = 2,
    ):
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5]
        self.num_classes = num_classes
        self.vehicle_indices = VEHICLE_INDICES
        self.reset()

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        """
        Process a batch of predicted and ground truth BEV segmentation maps.

        Parameters
        ----------
        output : torch.Tensor
            Predicted BEV segmentation heatmaps, shape [batch, 1, 200, 200], float32.
            Represents model logits or probabilities for vehicle presence.
        gt : torch.Tensor
            Ground truth BEV segmentation maps, shape [batch, 200, 200, 12], float32.
            Contains binary labels for 12 semantic classes.
        """
        pred_bev = output.detach().cpu()

        gt_bev = gt.detach().cpu()

        gt_bev = gt_bev.permute(0, 3, 1, 2)

        gt_vehicle_channels = gt_bev[:, self.vehicle_indices]
        gt_vehicle_mask = gt_vehicle_channels.max(dim=1, keepdim=False).values

        pred_probs = torch.sigmoid(pred_bev)
        pred_labels = (pred_probs > self.iou_thresholds[0]).float().squeeze(1)
        self.confusion_matrix += self._generate_matrix(gt_vehicle_mask, pred_labels)
