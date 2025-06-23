# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Collection

import torch

# podm comes from the object-detection-metrics pip package
from podm.metrics import BoundingBox

from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models.gear_guard_net.app import postprocess


class GearGuardNetEvaluator(DetectionEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
    ):
        self.threshhold = 0.7
        self.iou_thr = 0.5
        self.scale_x = 1 / image_width
        self.scale_y = 1 / image_height
        DetectionEvaluator.__init__(
            self, image_height, image_width, self.threshhold, self.iou_thr
        )

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]):
        image_ids, scales, paddings, all_bboxes, all_classes, all_num_boxes = gt

        for i in range(len(image_ids)):
            image_id = image_ids[i]
            bboxes = all_bboxes[i][: all_num_boxes[i].item()]
            classes = all_classes[i][: all_num_boxes[i].item()]
            if bboxes.numel() == 0:
                continue

            # Collect GT and prediction boxes
            gt_bb_entry = []
            for j in range(len(bboxes)):
                if classes[j] == 0 or classes[j] == 1:
                    gt_bb_entry.append(
                        BoundingBox.of_bbox(
                            image_id,
                            int(classes[j]),
                            bboxes[j][0].item(),
                            bboxes[j][1].item(),
                            bboxes[j][2].item(),
                            bboxes[j][3].item(),
                            1.0,
                        )
                    )

            output_single = [o[i : i + 1].permute(0, 2, 3, 1).detach() for o in output]
            result = postprocess(
                output_single, scales[i], paddings[i], self.threshhold, self.iou_thr
            )

            pd_bb_entry = []
            for item in result:
                class_id, x0, y0, x1, y1, score = item
                pd_bb_entry.append(
                    BoundingBox.of_bbox(
                        image_id,
                        class_id,
                        float(x0),
                        float(y0),
                        float(x1),
                        float(y1),
                        score,
                    )
                )

            # Compute mean average precision
            self._update_mAP(gt_bb_entry, pd_bb_entry)
