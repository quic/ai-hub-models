# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import torch

from qai_hub_models.evaluators.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.utils.image_processing import denormalize_coordinates_affine


class CenternetPoseEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for Centernet pose estimation models"""

    def __init__(
        self,
        decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ],
        max_dets: int = 100,
        in_vis_thre: float = 0.2,
    ):
        """
        decode:
            Function to decode the raw model outputs
            into detected objects/detections and keypoints.
        max_det (int):
            Maximum number of detections per image.
        """
        self.decode = decode
        self.max_dets = max_dets
        super().__init__(in_vis_thre)

    def add_batch(
        self,
        output: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | torch.Tensor
        ),
        gt: list[torch.Tensor],
    ) -> None:
        """Process a batch of Centernet model outputs and ground truth data.

        Parameters
        ----------
            output: Model predictions which can be :
                - hm (torch.Tensor): Heatmap with the shape of
                    [B, num_classes, H//4, W//4].
                - wh (torch.Tensor): Width/Height value with the
                    shape of [B, 2, H//4, W//4].
                - hps (torch.tensor): keypoint offsets relative to the object center
                    with the shape of [B, 2* num_joints, H//4, W//4].
                - reg (torch.Tensor): 2D regression value with the
                    shape of [B, 2, H//4, W//4].
                - hm_hp (torch.Tensor): Keypoint heatmap with the
                    shape of [B, num_joints, H//4, W//4].
                - hm_offset (torch.Tensor): heatmap offset with
                    the shape of [B, 2, H//4, W//4].
                where num_joints = 17, num_classes = 1.
            gt: Ground truth data containing:
                - image_ids: Tensor[int] of COCO image IDs [batch]
                - category_ids: Tensor[int] of category IDs [batch]
                - centers: Tensor[float] of bounding box centers [batch, 2]
                - scale: Tensor[float] of scale factors [batch, 2]
        """
        hm, wh, hps, reg, hm_hp, hm_offset = output
        image_ids, category_ids, center, scale = gt

        dets_pt = self.decode(hm, wh, hps, reg, hm_hp, hm_offset, self.max_dets)
        dets = dets_pt.detach().numpy()

        # Take top predicted heatmaps
        preds = dets[:, 0, 5:39].reshape(-1, 17, 2)
        maxvals = dets[:, 0, 4].reshape(-1, 1, 1).repeat(17, axis=1)

        for i in range(preds.shape[0]):
            preds[i] = denormalize_coordinates_affine(
                preds[i],
                center[i].numpy(),
                scale[i].numpy(),
                0,
                (hm.shape[2], hm.shape[3]),
            )

        self._store_predictions(preds, maxvals, image_ids, category_ids)
