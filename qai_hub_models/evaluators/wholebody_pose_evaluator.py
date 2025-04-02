# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from mmpose.codecs.utils import get_simcc_maximum
from xtcocotools.cocoeval import COCOeval

from qai_hub_models.datasets.cocowholebody import CocoWholeBodyDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.utils.pose import (
    BODY_SIGMAS,
    FACE_SIGMAS,
    FOOT_SIGMAS,
    LEFTHAND_SIGMAS,
    RIGHTHAND_SIGMAS,
)
from qai_hub_models.utils.printing import suppress_stdout


class WholeBodyPoseEvaluator(BaseEvaluator):
    """Evaluator for keypoint-based pose estimation using COCO-style mAP."""

    def __init__(self, image_height: int, image_width: int, in_vis_thre=0.2):
        """
        Args:
            coco_gt: COCO ground truth dataset.
        """
        self.reset()
        self.in_vis_thre = in_vis_thre
        self.coco_gt = CocoWholeBodyDataset().cocoGt
        self.body_num = 17
        self.foot_num = 6
        self.hand_num = 21
        self.face_num = 68
        self.total_kpts = (
            self.body_num + self.foot_num + 2 * self.hand_num + self.face_num
        )
        self.input_size = (image_width, image_height)

    def reset(self):
        """Resets the collected predictions."""
        self.predictions = []

    def add_batch(self, output: torch.Tensor, gt: list[torch.Tensor]):
        """
        Collects model predictions in COCO format and scales keypoints to original image space.
        """
        pred_x, pred_y = output
        batch_size = pred_x.shape[0]
        keypoints, scores = self.decode_output(pred_x.numpy(), pred_y.numpy())
        image_ids, category_ids, scale, bboxes = gt
        input_size_array = np.array(self.input_size, dtype=np.float32)
        cuts = (
            np.cumsum(
                [
                    0,
                    self.body_num,
                    self.foot_num,
                    self.face_num,
                    self.hand_num,
                    self.hand_num,
                ]
            )
            * 3
        )
        for idx in range(batch_size):
            keypoints_batch = keypoints[idx]  # Shape: (133, 2), NumPy
            bbox = bboxes[idx]
            scale_batch = scale[idx].numpy()
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            keypoints_batch = (
                keypoints_batch / input_size_array * scale_batch
                + center
                - 0.5 * scale_batch
            )
            keypoints_list = []
            for joint_idx in range(keypoints_batch.shape[0]):
                x_coordinate, y_coordinate = keypoints_batch[joint_idx]
                visibility_score = float(scores[idx][joint_idx])
                if visibility_score < self.in_vis_thre:
                    visibility_score = 0
                    x_coordinate, x_coordinate = 0.0, 0.0
                keypoints_list.extend(
                    [
                        x_coordinate,
                        y_coordinate,
                        2 if visibility_score > self.in_vis_thre else 0,
                    ]
                )

            # Verify keypoint count
            assert (
                len(keypoints_list) == self.total_kpts * 3
            ), f"Expected {self.total_kpts * 3}, got {len(keypoints_list)}"

            body_kpts = keypoints_list[cuts[0] : cuts[1]]
            foot_kpts = keypoints_list[cuts[1] : cuts[2]]
            face_kpts = keypoints_list[cuts[2] : cuts[3]]
            lefthand_kpts = keypoints_list[cuts[3] : cuts[4]]
            righthand_kpts = keypoints_list[cuts[4] : cuts[5]]

            # Scoring
            box_score = np.mean(scores[idx])
            kpt_score = 0.0
            valid_num = 0
            for n_jt in range(scores.shape[1]):
                t_s = scores[idx][n_jt]
                if t_s > self.in_vis_thre:
                    kpt_score += t_s
                    valid_num += 1
            if valid_num > 0:
                kpt_score /= valid_num
            final_score = kpt_score * box_score

            # Prediction dict
            pred_dict = {
                "image_id": int(image_ids[idx]),
                "category_id": int(category_ids[idx]),
                "keypoints": body_kpts,
                "foot_kpts": foot_kpts,
                "righthand_kpts": righthand_kpts,
                "lefthand_kpts": lefthand_kpts,
                "face_kpts": face_kpts,
                "center": center,
                "scale": scale_batch,
                "score": float(final_score),
            }
            self.predictions.append(pred_dict)

    def decode_output(
        self, pred_x: np.ndarray, pred_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from SimCC representations. The decoded
        coordinates are in the input image space.

        Args:
            encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x-axis
                and y-axis
            simcc_x (np.ndarray): SimCC label for x-axis
            simcc_y (np.ndarray): SimCC label for y-axis

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """
        keypoints, scores = get_simcc_maximum(pred_x, pred_y)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]
        keypoints /= 2.0

        return keypoints, scores

    def get_coco_mAP(self) -> dict[str, Any]:
        """
        Computes COCO-style mAP using COCOfooteval.
        Returns:
            A dictionary with AP values (mAP, AP@0.5, etc.).
        """

        valid_image_ids = set(self.coco_gt.getImgIds())
        pred_image_ids = {p["image_id"] for p in self.predictions}

        res = copy.deepcopy(self.predictions)
        sigmas = np.array(
            BODY_SIGMAS + FOOT_SIGMAS + FACE_SIGMAS + LEFTHAND_SIGMAS + RIGHTHAND_SIGMAS
        )
        with suppress_stdout():
            coco_det = self.coco_gt.loadRes(res)
            coco_eval = COCOeval(
                self.coco_gt, coco_det, "keypoints_wholebody", sigmas, use_area=True
            )
            coco_eval.params.catIds = [1]
            coco_eval.params.imgIds = list(valid_image_ids & pred_image_ids)
            coco_eval.params.useSegm = None
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
