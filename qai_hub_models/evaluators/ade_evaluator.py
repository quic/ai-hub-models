# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn.functional as F

from qai_hub_models.evaluators.segmentation_evaluator import SegmentationOutputEvaluator


class AdeEvaluator(SegmentationOutputEvaluator):
    """Evaluator for comparing segmentation output against ground truth"""

    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):

        # Handle tuple outputs
        if isinstance(output, tuple):
            output = output[0]

        pred_masks = F.interpolate(
            input=output,
            size=gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Convert to class predictions
        pred_mask = torch.argmax(pred_masks, 1)
        output = pred_mask.cpu()
        assert gt.shape == output.shape
        self.confusion_matrix += self._generate_matrix(gt, output)
