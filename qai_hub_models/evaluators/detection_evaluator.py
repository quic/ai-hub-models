# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Collection

import torch

# podm comes from the object-detection-metrics pip package
from podm.metrics import BoundingBox, MetricPerClass, get_pascal_voc_metrics

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
        use_nms: bool = True,
        score_threshold: float = 0.9,
    ):
        self.reset()
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.scale_x = 1 / image_height
        self.scale_y = 1 / image_width
        self.use_nms = use_nms
        self.score_threshold = score_threshold

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]):
        """
        Args:
            output: A tuple of tensors containing the predicted bounding boxes, scores, and class indices.
                - bounding boxes with shape (batch_size, num_candidate_boxes, 4)
                - The 4 should be normalized (x, y, w, h)
                - scores with shape (batch_size, num_candidate_boxes)
                - class predictions with shape (batch_size, num_candidate_boxes)
            gt: A tuple of tensors containing the ground truth bounding boxes and other metadata.
                - image_ids of shape (batch_size,)
                - image heights of shape (batch_size,)
                - image widths of shape (batch_size,)
                - bounding boxes of shape (batch_size, max_boxes, 4)
                - The 4 should be normalized (x, y, w, h)
                - classes of shape (batch_size, max_boxes)
                - num nonzero boxes for each sample of shape (batch_size,)

        Note:
            The `output` and `gt` tensors should be of the same length, i.e., the same batch size.
        """
        image_ids, _, _, all_bboxes, all_classes, all_num_boxes = gt
        pred_boxes, pred_scores, pred_class_idx = output
        for i in range(len(image_ids)):
            image_id = image_ids[i]
            bboxes = all_bboxes[i][: all_num_boxes[i].item()]
            classes = all_classes[i][: all_num_boxes[i].item()]
            if bboxes.numel() == 0:
                continue
            curr_pred_box = pred_boxes[i : i + 1]
            curr_pred_score = pred_scores[i : i + 1]
            curr_pred_class = pred_class_idx[i : i + 1]

            if self.use_nms:
                (curr_pred_box, curr_pred_score, curr_pred_class,) = batched_nms(
                    self.nms_iou_threshold,
                    self.nms_score_threshold,
                    curr_pred_box,
                    curr_pred_score,
                    curr_pred_class,
                )

            # Collect GT and prediction boxes
            gt_bb_entry = [
                BoundingBox.of_bbox(
                    image_id, cat, bbox[0], bbox[1], bbox[2], bbox[3], 1.0
                )
                for cat, bbox in zip(classes.tolist(), bboxes.tolist())
            ]

            pd_bb_entry = [
                BoundingBox.of_bbox(
                    image_id,
                    pred_cat,
                    float(pred_bbox[0]) * self.scale_x,
                    float(pred_bbox[1]) * self.scale_y,
                    float(pred_bbox[2]) * self.scale_x,
                    float(pred_bbox[3]) * self.scale_y,
                    float(pred_score),
                )
                for pred_cat, pred_score, pred_bbox in zip(
                    curr_pred_class[0].tolist(),
                    curr_pred_score[0].tolist(),
                    curr_pred_box[0].tolist(),
                )
            ]

            # Mask and threshold predictions
            if not self.use_nms:
                # Create a new list that includes only the predictions with scores above the threshold
                pd_bb_entry = [b for b in pd_bb_entry if b.score > self.score_threshold]

            self._update_mAP(gt_bb_entry, pd_bb_entry)

    def reset(self):
        self.gt_bb = []
        self.pd_bb = []
        self.results = {}

    def _update_mAP(self, gt_bb_entry, pd_bb_entry):
        self.gt_bb += gt_bb_entry
        self.pd_bb += pd_bb_entry

        self.results = get_pascal_voc_metrics(
            self.gt_bb, self.pd_bb, self.nms_iou_threshold
        )

        self.mAP = MetricPerClass.mAP(self.results)

    def get_accuracy_score(self):
        return self.mAP

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} mAP"
