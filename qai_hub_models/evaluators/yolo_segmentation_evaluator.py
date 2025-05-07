# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from torchmetrics.detection import MeanAveragePrecision
from ultralytics.utils.ops import process_mask

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.utils.bounding_box_processing import batched_nms


class YoloSegmentationOutputEvaluator(BaseEvaluator):
    """Evaluator for comparing Instance Segmentation output against ground truth."""

    def __init__(
        self,
        image_height,
        image_width,
        nms_score_threshold: float = 0.001,
        nms_iou_threshold: float = 0.7,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.reset()

    def reset(self):
        self.preds = []
        self.expected = []

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        """
        gt should be a tuple of tensors with the following tensors:
            - mask of shape (batch_size, max_boxes, height, width)
              - The 4 should be normalized (x, y, w, h)
            - classes of shape (batch_size, max_boxes)
            - num nonzero boxes for each sample of shape (batch_size,)

        output should be a tuple of tensors with the following tensors:
            - bounding boxes with shape (batch_size, num_preds, 4)
              - The 4 should be normalized (x, y, w, h)
            - scores with shape (batch_size, num_preds)
            - class predictions with shape (batch_size, num_preds)
            - masks with shape [batch_size, num_preds, 32]
            - protos: with shape[batch_size, 32, mask_h, mask_w]
              - multiply masks and protos to generate output masks.
        """

        pred_boxes, pred_scores, pred_masks, pred_class_idx, proto = output
        gt_mask, gt_label, num_box = gt

        # Non Maximum Suppression on each batch
        pred_boxes, pred_scores, pred_class_idx, pred_masks = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes,
            pred_scores,
            pred_class_idx,
            pred_masks,
        )

        for batch_idx in range(len(pred_masks)):
            # Process mask and upsample to input shape
            pred_masks[batch_idx] = process_mask(
                proto[batch_idx],
                pred_masks[batch_idx],
                pred_boxes[batch_idx],
                (self.image_height, self.image_width),
                upsample=True,
            )

            self.preds.append(
                {
                    "masks": pred_masks[batch_idx].to(torch.uint8),
                    "scores": pred_scores[batch_idx],
                    "labels": pred_class_idx[batch_idx].to(torch.uint8),
                }
            )
            self.expected.append(
                {
                    "masks": gt_mask[batch_idx][: num_box[batch_idx]],
                    "labels": gt_label[batch_idx][: num_box[batch_idx]],
                }
            )

    def get_accuracy_score(self) -> float:
        # Calculate mAP_mask[0.5-0.95]
        metric = MeanAveragePrecision(iou_type="segm")
        metric.update(self.preds, self.expected)
        results = metric.compute()
        mAP = results["map"]
        return mAP

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} mAP"
