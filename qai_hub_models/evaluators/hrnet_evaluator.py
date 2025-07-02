# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.evaluators.utils.pose import get_final_preds


class HRNetPoseEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for HRNet pose estimation models"""

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        gt: list[torch.Tensor],
    ) -> None:
        """Process a batch of HRNet model outputs and ground truth data.

        Args:
            output: Model predictions which can be :
                   - A tuple containing (heatmaps,) [batch, joints, H, W]
            gt: Ground truth data containing:
                - image_ids: Tensor[int] of COCO image IDs [batch]
                - category_ids: Tensor[int] of category IDs [batch]
                - centers: Tensor[float] of bounding box centers [batch, 2]
                - scale: Tensor[float] of scale factors [batch, 2]
        """
        output = output[0] if isinstance(output, tuple) else output
        image_ids, category_ids, center, scale = gt
        preds, maxvals = get_final_preds(output.detach().cpu().numpy(), center, scale)
        self._store_predictions(preds, maxvals, image_ids, category_ids)
