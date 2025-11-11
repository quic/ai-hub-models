# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Collection

import torch

from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator


class CenternetDetectionEvaluator(DetectionEvaluator):
    """Evaluator for Centernet detection tasks."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        decode: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, bool, int], torch.Tensor
        ],
        max_dets: int = 100,
        cat_spec_wh: bool = False,
        nms_iou_threshold: float | None = None,
        score_threshold: float | None = None,
        mAP_default_low_iOU: float | None = None,
        mAP_default_high_iOU: float | None = None,
        mAP_default_increment_iOU: float | None = None,
    ):
        """
        decode:
            Function to decode the raw model outputs
            into detected objects/detections.
        cat_spec_wh (bool):
            If True, indicates that the `wh` tensoris category-specific
            (i.e., its channel dimension is `2 * num_classes`). If False,
            `wh` is not category-specific. Defaults to False.
        max_det (int):
            Maximum number of detections per image.
        """
        self.decode = decode
        self.max_dets = max_dets
        self.cat_spec_wh = cat_spec_wh
        super().__init__(
            image_height=image_height,
            image_width=image_width,
            nms_iou_threshold=nms_iou_threshold,
            score_threshold=score_threshold,
            mAP_default_low_iOU=mAP_default_low_iOU,
            mAP_default_high_iOU=mAP_default_high_iOU,
            mAP_default_increment_iOU=mAP_default_increment_iOU,
        )

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]):
        """
        Parameters
        ----------
            output: A tuple of tensors containing;
                - hm (torch.Tensor): Heatmap with the shape of
                    [B, num_classes, H//4, W//4].
                - wh (torch.Tensor): Width/Height value with the
                    shape of [B, 2, H//4, W//4].
                - reg (torch.Tensor): 2D regression value with the
                    shape of [B, 2, H//4, W//4].
                where num_classes = 80.

            gt: A tuple of tensors containing the ground truth bounding boxes and other metadata.
                - image_ids of shape (batch_size,)
                - image heights of shape (batch_size,)
                - image widths of shape (batch_size,)
                - bounding boxes of shape (batch_size, max_boxes, 4) - 4 order is (x1, y1, x2, y2)
                - classes of shape (batch_size, max_boxes)
                - num nonzero boxes for each sample of shape (batch_size,)

        Note:
            The `output` and `gt` tensors should be of the same length, i.e., the same batch size.
        """
        hm, wh, reg = output
        dets = self.decode(hm, wh, reg, self.cat_spec_wh, self.max_dets)
        dets = dets.detach()
        pred_boxes = dets[:, :, :4]
        x1 = pred_boxes[:, :, 0]
        y1 = pred_boxes[:, :, 1]
        x2 = pred_boxes[:, :, 2]
        y2 = pred_boxes[:, :, 3]

        # Apply min/max operations to swap if necessary
        pred_boxes[:, :, 0] = torch.min(x1, x2)
        pred_boxes[:, :, 2] = torch.max(x1, x2)
        pred_boxes[:, :, 1] = torch.min(y1, y2)
        pred_boxes[:, :, 3] = torch.max(y1, y2)
        pred_scores = dets[:, :, 4]

        pred_class_idx = dets[:, :, 5].int()
        super().add_batch((pred_boxes, pred_scores, pred_class_idx), gt)
