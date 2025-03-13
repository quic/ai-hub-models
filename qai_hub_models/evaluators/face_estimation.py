# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from xtcocotools.cocoeval import COCOeval

from qai_hub_models.datasets.coco_face import CocoFaceDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.facemap_3dmm.utils import project_landmark
from qai_hub_models.utils.printing import suppress_stdout


class FacemapFaceEstimationEvaluator(BaseEvaluator):
    """Evaluator for keypoint-based Face landmark estimation using COCO-style mAP."""

    def __init__(self, height, width):
        """
        Args:
            coco_gt: COCO ground truth dataset.
        """
        self.reset()
        self.height = height
        self.width = width
        self.coco_gt = CocoFaceDataset().cocoGt

    def reset(self):
        """Resets the collected predictions."""
        self.predictions = []

    def add_batch(self, output: torch.Tensor, gt: list[torch.Tensor]):
        """
        Collects model predictions in COCO format, handling both single and batched keypoints.

        Args:
            output: 3DMM model parameters for facial landmark reconstruction with shape of [batch, 265]
            gt_data: Ground truth data from dataset containing (image_id, category_id, bbox).
        """
        batch_size = output.shape[0]

        # Extract batched ground truth values
        image_ids, category_ids, bbox = gt

        for idx in range(batch_size):
            face_landmark = project_landmark(output[idx])

            x0, y0, x1, y1 = bbox[idx]
            bbox_height = y1 - y0 + 1
            bbox_width = x1 - x0 + 1

            # resize the landmark to original size
            face_landmark[:, 0] = (
                face_landmark[:, 0] + self.width / 2
            ) * bbox_width / self.width + x0
            face_landmark[:, 1] = (
                face_landmark[:, 1] + self.height / 2
            ) * bbox_height / self.height + y0

            # added score as 1
            face_landmark = torch.concat([face_landmark, torch.ones(68, 1)], dim=1)
            face_landmark = list(face_landmark.reshape(-1))

            # cocoeval API requires 17 body keypoints
            body_kpts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            # Store prediction in correct format
            pred_dict = {
                "image_id": int(image_ids[idx]),
                "category_id": int(category_ids[idx]),
                "keypoints": body_kpts,
                "face_kpts": face_landmark,
                "score": float(1),
            }

            self.predictions.append(pred_dict)

    def get_coco_mAP(self) -> dict[str, Any]:
        """
        Computes COCO-style mAP using COCOeval.

        Returns:
            A dictionary with AP values (mAP, AP@0.5, etc.).
        """
        pred_image_ids = [p["image_id"] for p in self.predictions]

        # https://github.com/nerminsamet/HPRNet/blob/3ffba9497cdc7b935bf9863f93decfde4b9b7eae/src/tools/coco_body_eval/myeval_face.py#L169
        sigmas = (
            np.array(
                [
                    0.42,
                    0.43,
                    0.44,
                    0.43,
                    0.4,
                    0.35,
                    0.31,
                    0.25,
                    0.2,
                    0.23,
                    0.29,
                    0.32,
                    0.37,
                    0.38,
                    0.43,
                    0.41,
                    0.45,
                    0.13,
                    0.12,
                    0.11,
                    0.11,
                    0.12,
                    0.12,
                    0.11,
                    0.11,
                    0.13,
                    0.15,
                    0.09,
                    0.07,
                    0.07,
                    0.07,
                    0.12,
                    0.09,
                    0.08,
                    0.16,
                    0.10,
                    0.17,
                    0.11,
                    0.09,
                    0.11,
                    0.09,
                    0.07,
                    0.13,
                    0.08,
                    0.11,
                    0.12,
                    0.10,
                    0.34,
                    0.08,
                    0.08,
                    0.09,
                    0.08,
                    0.08,
                    0.07,
                    0.10,
                    0.08,
                    0.09,
                    0.09,
                    0.09,
                    0.07,
                    0.07,
                    0.08,
                    0.11,
                    0.08,
                    0.08,
                    0.08,
                    0.10,
                    0.08,
                ]
            )
            / 10.0
        )

        res = copy.deepcopy(self.predictions)
        with suppress_stdout():
            coco_dt = self.coco_gt.loadRes(res)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "keypoints_face", sigmas)
            coco_eval.params.useSegm = None
            coco_eval.params.imgIds = pred_image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        return {"AP": coco_eval.stats[0], "AP@.5": coco_eval.stats[1]}

    def get_accuracy_score(self) -> float:
        """Returns the overall mAP score."""
        return self.get_coco_mAP()["AP"]

    def formatted_accuracy(self) -> str:
        """Formats the mAP score for display."""
        results = self.get_coco_mAP()
        return f"mAP: {results['AP']:.3f}, AP@.5: {results['AP@.5']:.3f}"
