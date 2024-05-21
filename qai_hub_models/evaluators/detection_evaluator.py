# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Collection

import torch
from podm.metrics import (  # type: ignore
    BoundingBox,
    MetricPerClass,
    get_pascal_voc_metrics,
)

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.utils.bounding_box_processing import batched_nms


class DetectionEvaluator(BaseEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
    ):
        self.reset()
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.scale_x = 1 / image_height
        self.scale_y = 1 / image_width

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]):
        # This evaluator supports 1 output tensor at a time.
        image_id, _, _, bboxes, classes = gt
        pred_boxes, pred_scores, pred_class_idx = output

        if bboxes.numel() == 0:
            return

        # The number of boxes can be variable, so dataloader doesn't like shapes
        # mismatching across samples in the batch.
        assert bboxes.shape[0] == 1, "Detection evaluator only supports batch size 1."
        bboxes = bboxes.squeeze(0)
        classes = classes.squeeze(0)

        # Seeing memory issues, initentionally deleting these variables to free memory.
        del gt
        del output

        # Reuse NMS utility
        (
            after_nms_pred_boxes,
            after_nms_pred_scores,
            after_nms_pred_class_idx,
        ) = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes,
            pred_scores,
            pred_class_idx,
        )

        del pred_boxes
        del pred_scores
        del pred_class_idx

        # Collect GT and prediction boxes
        gt_bb_entry = [
            BoundingBox.of_bbox(image_id, cat, *bbox, 1.0)
            for cat, bbox in zip(classes.tolist(), bboxes.tolist())
        ]
        del classes
        del bboxes

        pd_bb_entry = [
            BoundingBox.of_bbox(
                image_id,
                pred_cat,
                pred_bbox[0] * self.scale_x,
                pred_bbox[1] * self.scale_y,
                pred_bbox[2] * self.scale_x,
                pred_bbox[3] * self.scale_y,
                pred_score,
            )
            for pred_cat, pred_score, pred_bbox in zip(
                after_nms_pred_class_idx[0].tolist(),
                after_nms_pred_scores[0].tolist(),
                after_nms_pred_boxes[0].tolist(),
            )
        ]

        del after_nms_pred_boxes
        del after_nms_pred_scores
        del after_nms_pred_class_idx

        # Compute mean average precision
        self._update_mAP(gt_bb_entry, pd_bb_entry)

    def reset(self):
        self.gt_bb = []
        self.pd_bb = []
        self.results = {}

    def _update_mAP(self, gt_bb_entry, pd_bb_entry):
        self.gt_bb += gt_bb_entry
        self.pd_bb += pd_bb_entry

        del gt_bb_entry
        del pd_bb_entry
        self.results = get_pascal_voc_metrics(
            self.gt_bb, self.pd_bb, self.nms_iou_threshold
        )
        self.mAP = MetricPerClass.mAP(self.results)

    def get_accuracy_score(self):
        return self.mAP

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score()} mAP"
