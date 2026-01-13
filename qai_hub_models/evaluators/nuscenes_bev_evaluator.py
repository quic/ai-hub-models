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
        iou_thresholds: float = 0.5,
        min_visibility: int = 2,
    ):
        self.iou_thresholds = torch.tensor([iou_thresholds])
        self.vehicle_indices = VEHICLE_INDICES
        self.min_visibility = min_visibility
        self.reset()

    def reset(self):
        """Reset evaluation metrics."""
        self.tp = torch.zeros_like(self.iou_thresholds)
        self.fp = torch.zeros_like(self.iou_thresholds)
        self.fn = torch.zeros_like(self.iou_thresholds)

    def add_batch(self, output: torch.Tensor, gt: tuple[torch.Tensor, torch.Tensor]):
        """
        Process a batch of predicted and ground truth BEV segmentation maps.

        Parameters
        ----------
        output
            Predicted BEV segmentation heatmaps with shape [batch, 1, 200, 200], float32.
            Represents model logits or probabilities for vehicle presence.
        gt
            Ground truth BEV segmentation maps
                torch.Tensor of shape [batch, 200, 200, 12], float32.
                Contains binary labels for 12 semantic classes.
            visibility
                torch.Tensor of shape [H_bev, W_bev] as uint8
                in range [1-255], higher value = more visible
        """
        pred_bev = output.detach().cpu()
        gt_bev, visibility = gt

        gt_bev = gt_bev.permute(0, 3, 1, 2)

        gt_vehicle_channels = gt_bev[:, self.vehicle_indices]
        gt_vehicle_mask = gt_vehicle_channels.max(dim=1, keepdim=False).values

        pred_probs = torch.sigmoid(pred_bev)

        # Create mask for sufficiently visible regions
        vis_mask = visibility >= self.min_visibility
        vis_mask = vis_mask.unsqueeze(1)

        # Apply mask before flattening
        pred_probs_filtered = pred_probs[vis_mask]
        gt_vehicle_mask_filtered = gt_vehicle_mask.unsqueeze(1)[vis_mask]

        gt_labels = gt_vehicle_mask_filtered.bool()
        pred_binary = pred_probs_filtered[:, None] >= self.iou_thresholds[None, :]
        gt_labels = gt_labels[:, None]

        self.tp += (pred_binary & gt_labels).sum(0)
        self.fp += (pred_binary & ~gt_labels).sum(0)
        self.fn += (~pred_binary & gt_labels).sum(0)

    def get_accuracy_score(self) -> float:
        return float(self.tp / (self.tp + self.fp + self.fn + 1e-7))

    def formatted_accuracy(self) -> str:
        return f"mAP@{float(self.iou_thresholds):.2f}: {self.get_accuracy_score():.4f}"
