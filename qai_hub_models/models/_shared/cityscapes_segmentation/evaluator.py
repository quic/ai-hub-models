# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch.nn.functional as F
from torch import Tensor

from qai_hub_models.evaluators.image_evaluator import SegmentationOutputEvaluator


class CityscapesSegmentationEvaluator(SegmentationOutputEvaluator):
    """
    Evaluates the output of Cityscapes semantics segmentation.
    """

    def add_batch(self, output: Tensor, gt: Tensor):
        output_match_size = F.interpolate(output, gt.shape[1:3], mode="bilinear")
        output_class = output_match_size.argmax(1).cpu()
        return super().add_batch(output_class, gt)

    def get_accuracy_score(self) -> float:
        return super().Mean_Intersection_over_Union()
