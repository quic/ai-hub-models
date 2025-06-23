# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np

# podm comes from the object-detection-metrics pip package
from podm.metrics import BoundingBox

from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models.gear_guard_net.app import postprocess


class PpeEvaluator(DetectionEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
    ):
        self.threshhold = 0.7
        self.iou_thr = 0.5
        self.scale_x = 1 / image_width
        self.image_width = image_width
        self.image_height = image_height
        self.scale_y = 1 / image_height
        DetectionEvaluator.__init__(
            self, image_height, image_width, self.threshhold, self.iou_thr
        )

    def add_batch(self, output, gt):
        image_ids, categories, paddings, scales, gt_bboxes, person_bboxes = gt
        batch_size = len(image_ids)

        for i in range(batch_size):
            gt_bbox = np.array(gt_bboxes[i])
            image_id = image_ids[i]
            gt_class = np.array([categories[i]])

            gt_entries = []
            gt_entries.append(
                BoundingBox.of_bbox(
                    image_id,
                    1 if gt_class[0] == 5 else 0,
                    gt_bbox[0],
                    gt_bbox[1],
                    gt_bbox[2],
                    gt_bbox[3],
                    1.0,
                )
            )

            # Process predictions
            output_single = [o[i : i + 1].permute(0, 2, 3, 1).detach() for o in output]
            result = postprocess(
                output_single, scales[i], paddings[i], self.threshhold, self.iou_thr
            )
            pred_entries = []
            if result.size > 0:
                for res in result:
                    class_id, x, y, w, h, score = res
                    px1, py1, _, _ = person_bboxes[i]
                    pred_entries.append(
                        BoundingBox.of_bbox(
                            image_id,
                            int(class_id),
                            x + px1,
                            y + py1,
                            w + px1,
                            h + py1,
                            float(score),
                        )
                    )
            self._update_mAP(gt_entries, pred_entries)
