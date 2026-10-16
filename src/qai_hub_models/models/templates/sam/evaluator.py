# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MEAN_IOU, MetricMetadata


class SAMMaskIoUEvaluator(BaseEvaluator):
    """Mean IoU between SAM's predicted mask and the ground-truth instance mask."""

    def __init__(self, mask_threshold: float = 0.0) -> None:
        self.mask_threshold = mask_threshold
        self.reset()

    def reset(self) -> None:
        self.total_iou = 0.0
        self.num_samples = 0

    def add_batch(
        self,
        output: tuple[torch.Tensor, ...],
        gt: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        """
        Parameters
        ----------
        output
            (masks, scores) from the SAM decoder. masks are low-res logits
            with shape [B, 1, h, w] covering the model input frame.
            Further tensors are ignored.
        gt
            Binary instance masks with shape [B, H, W] in the model input frame,
            either as a tensor or wrapped in a 1-tuple.
        """
        gt_mask = (gt if isinstance(gt, torch.Tensor) else gt[0]).bool()
        logits = F.interpolate(
            output[0].float(),
            size=gt_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = logits.squeeze(1) > self.mask_threshold
        intersection = (pred_mask & gt_mask).flatten(1).sum(1).float()
        union = (pred_mask | gt_mask).flatten(1).sum(1).float()
        self.total_iou += float((intersection / union.clamp(min=1)).sum())
        self.num_samples += gt_mask.shape[0]

    def get_accuracy_score(self) -> float:
        return 100 * self.total_iou / max(self.num_samples, 1)

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} mIOU"

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_IOU
