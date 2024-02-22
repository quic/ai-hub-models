# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from torch import Tensor

from qai_hub_models.evaluators.image_evaluator import SegmentationOutputEvaluator


class DeepLabV3Evaluator(SegmentationOutputEvaluator):
    """
    Evaluates the output of DeepLabV3Plus

    Expected data format for this evaluator:
    * output has the same shape & meaning as output of any deeplabV3 forward() function.
    * gt is argmax'd on the first dimension (see add_batch).
    """

    def add_batch(self, output: Tensor, gt: Tensor):
        output = output.argmax(1).cpu()
        return super().add_batch(output, gt)

    def get_accuracy_score(self) -> float:
        return super().Mean_Intersection_over_Union()
