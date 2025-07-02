# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.evaluators.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.models.litehrnet.app import refine_and_transform_keypoints


class LiteHRNetPoseEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for LiteHRNet pose estimation models"""

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt: list[torch.Tensor],
    ) -> None:
        """
        Processes LiteHRNet outputs (keypoints, scores, heatmaps) and converts them to
        COCO-format keypoint predictions with refinement using heatmaps.

        Args:
            output: Tuple containing:
                   - keypoints: Tensor[float] of predicted keypoints [batch, 17, 2]
                   - pred_scores:Tensor[float] of confidence scores [batch, 17, 1]
                   - heatmaps: Tensor[float] of heatmaps [batch, 17, 64 , 48]
            gt: Ground truth data containing:
                - image_ids: Tensor[int] of COCO image IDs [batch]
                - category_ids: Tensor[int] of category IDs [batch]
                - scale: Tensor[float] of scale factors [batch, 2]
                - bboxes: Tensor[float] of bounding boxes [batch, 4] in (x1,y1,x2,y2) format
        """
        image_ids, category_ids, scale, bboxes = gt
        keypoints, pred_scores, heatmaps = output
        batch_size = keypoints.shape[0]
        preds = keypoints.cpu().numpy()
        maxvals = pred_scores.detach().cpu().numpy()
        for idx in range(batch_size):
            preds_batch = refine_and_transform_keypoints(
                np.expand_dims(preds[idx], axis=0),
                heatmaps[idx].unsqueeze(0),
                bboxes[idx],
                scale[idx],
            )
            preds[idx] = preds_batch[0]
        self._store_predictions(preds, maxvals, image_ids, category_ids)
