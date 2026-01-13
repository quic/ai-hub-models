# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Collection

import torch

# podm comes from the object-detection-metrics pip package
from podm.metrics import BoundingBox

from qai_hub_models.evaluators.detection_evaluator import mAPEvaluator
from qai_hub_models.models.foot_track_net.app import postprocess


class FootTrackNetEvaluator(mAPEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(self):
        super().__init__()
        self.threshhold = [0.1, 0.1, 0.1]
        self.iou_thr = [0.2, 0.5, 0.5]

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]):
        """
        Add a batch of predictions and ground truth for evaluation.

        Parameters
        ----------
        output
            pred_boxes
                Shape (batch_size, num_candidate_boxes, 4) in normalized (x, y, w, h) format.
            pred_scores
                Shape (batch_size, num_candidate_boxes).
            pred_class_idx
                Shape (batch_size, num_candidate_boxes).
        gt
            image_ids
                Shape (batch_size,).
            image_heights
                Shape (batch_size,).
            image_widths
                Shape (batch_size,).
            bounding_boxes
                Shape (batch_size, max_boxes, 4) in normalized (x, y, w, h) format.
            classes
                Shape (batch_size, max_boxes).
            num_boxes
                Number of nonzero boxes for each sample, shape (batch_size,).
        """
        image_ids, _, _, all_bboxes, all_classes, all_num_boxes = gt
        output = list(output)

        for i in range(len(image_ids)):
            output_i = (
                output[0][i : i + 1],
                output[1][i : i + 1],
                output[2][i : i + 1],
                output[3][i : i + 1],
            )
            face_result, person_result = postprocess(
                *output_i, self.threshhold, self.iou_thr
            )

            image_id = image_ids[i]
            bboxes = all_bboxes[i][: int(all_num_boxes[i].item())]
            classes = all_classes[i][: int(all_num_boxes[i].item())]
            if bboxes.numel() == 0:
                continue

            # Collect GT and prediction boxes
            gt_bb_entry = [
                BoundingBox.of_bbox(
                    image_id,
                    int(classes[j]),
                    bboxes[j][0].item(),
                    bboxes[j][1].item(),
                    bboxes[j][2].item(),
                    bboxes[j][3].item(),
                    1.0,
                )
                for j in range(len(bboxes))
                if classes[j] == 0 or classes[j] == 1
            ]

            pd_bb_entry = [
                BoundingBox.of_bbox(
                    image_id,
                    0,
                    float(item.x),
                    float(item.y),
                    float(item.r),
                    float(item.b),
                    item.score,
                )
                for item in face_result
            ] + [
                BoundingBox.of_bbox(
                    image_id,
                    1,
                    float(item.x),
                    float(item.y),
                    float(item.r),
                    float(item.b),
                    item.score,
                )
                for item in person_result
            ]

            self.store_bboxes_for_eval(gt_bb_entry, pd_bb_entry)
