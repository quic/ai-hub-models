# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class DepthEvaluator(BaseEvaluator):
    """Evaluator for tracking accuracy of a Depth Estimation Model."""

    def __init__(self):
        self.reset()
        self.out = []

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        output = F.interpolate(
            output,
            size=gt.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        output = output.squeeze(1)

        assert gt.shape == output.shape
        # max depth is 10 in toolbox_nyu_depth_v2
        max_depth = 10.0
        mask = (gt > 0) & (gt < max_depth)

        # transform predicted disparity to aligned depth
        gt_disparity = torch.zeros_like(gt)
        gt_disparity[mask == 1] = 1.0 / gt[mask == 1]

        scale, shift = self.compute_scale_and_shift(output, gt_disparity, mask)
        output_aligned = scale.view(-1, 1, 1) * output + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / max_depth
        output_aligned[output_aligned < disparity_cap] = disparity_cap

        output_depth = 1.0 / output_aligned

        # calculate error
        err = torch.zeros_like(output_depth, dtype=torch.float)
        err[mask == 1] = torch.max(
            output_depth[mask == 1] / gt[mask == 1],
            gt[mask == 1] / output_depth[mask == 1],
        )
        err[mask == 1] = (err[mask == 1] < 1.25).float()

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))
        self.out.append(100 * torch.mean(p))

    def reset(self):
        self.out = []

    def get_accuracy_score(self) -> float:
        return np.mean(self.out)

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} δ1"

    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1
