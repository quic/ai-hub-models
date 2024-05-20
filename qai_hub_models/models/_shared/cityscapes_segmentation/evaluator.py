# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch.nn.functional as F
from torch import Tensor

from qai_hub_models.evaluators.segmentation_evaluator import SegmentationOutputEvaluator


class CityscapesSegmentationEvaluator(SegmentationOutputEvaluator):
    """
    Evaluates the output of Cityscapes semantics segmentation.
    """

    def add_batch(self, output: Tensor, gt: Tensor):
        output_match_size = F.interpolate(output, gt.shape[1:3], mode="bilinear")
        return super().add_batch(output_match_size, gt)
