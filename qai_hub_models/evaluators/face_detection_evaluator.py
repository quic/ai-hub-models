# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Collection

import torch
from podm.metrics import BoundingBox

from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models.face_det_lite.utils import detect


class FaceDetectionEvaluator(DetectionEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
    ):
        self.scale_x = 1 / image_width
        self.scale_y = 1 / image_height
        threshold = 0.55
        nms_iou = -1
        DetectionEvaluator.__init__(self, image_height, image_width, threshold, nms_iou)

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]):
        """
        gt should be a tuple of tensors (image_id, category_id, bbox).

        output should be a tuple of tensors with the following tensors:
            heatmap: (N,C,H,W) the heatmap for the person/face detection.
            bbox: (N,C*4,H,W) the bounding box coordinate as a map.
            landmark: (N,C*10,H,W) the coordinates of landmarks as a map.
        """

        hm, box, landmark = output

        # Extract batched ground truth values
        image_ids, category_ids, gt_bbox = gt

        for i in range(len(image_ids)):
            dets = detect(
                hm[i].unsqueeze(0),
                box[i].unsqueeze(0),
                landmark[i].unsqueeze(0),
                threshold=0.55,
                nms_iou=-1,
                stride=8,
            )
            image_id = image_ids[i]
            bboxes = gt_bbox[i]

            # Collect GT and prediction boxes
            gt_bb_entry = []
            gt_bb_entry.append(
                BoundingBox.of_bbox(
                    image_id,
                    int(0),
                    bboxes[0].item(),
                    bboxes[1].item(),
                    bboxes[2].item(),
                    bboxes[3].item(),
                    1.0,
                )
            )

            pd_bb_entry = []
            for item in dets:
                pd_bb_entry.append(
                    BoundingBox.of_bbox(
                        image_id,
                        0,
                        float(item.x) * self.scale_x,
                        float(item.y) * self.scale_y,
                        float(item.r) * self.scale_x,
                        float(item.b) * self.scale_y,
                        item.score,
                    )
                )

            # Compute mean average precision
            self._update_mAP(gt_bb_entry, pd_bb_entry)
