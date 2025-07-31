# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.evaluators.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.models.openpose.app import getKeypointsFromPredictions
from qai_hub_models.utils.image_processing import (
    apply_affine_to_coordinates,
    compute_affine_transform,
)

# Mapping OpenPose keypoint indices to COCO indices (Neck is ignored to match coco groundtruth)
OPENPOSE_TO_COCO = [
    0,  # Nose
    0,  # Neck (ignored)
    6,  # RShoulder
    8,  # RElbow
    10,  # RWrist
    5,  # LShoulder
    7,  # LElbow
    9,  # LWrist
    12,  # RHip
    14,  # RKnee
    16,  # RAnkle
    11,  # LHip
    13,  # LKnee
    15,  # LAnkle
    2,  # REye
    1,  # LEye
    4,  # REar
    3,  # LEar
]


class OpenPoseEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for OpenPose models on COCO-Body dataset"""

    def __init__(self, height, width, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_height = height
        self.input_width = width
        self.min_keypoints = 4
        self.min_score = 0.1

    def add_batch(
        self, output: tuple[torch.Tensor, torch.Tensor], gt: list[torch.Tensor]
    ):
        """
        Processes OpenPose outputs (PAFs and heatmaps) and converts them to
        COCO-format keypoint predictions

        Args:
            output: Tuple containing :
               - paf: Tensor[float] of part affinity fields [batch, 2, H, W]
               - heatmaps: Tensor[float] of keypoint heatmaps [batch, 19, H, W]
            gt list[torch.Tensor]:
            - image_ids: Tensor[int] of COCO image IDs [batch]
            - category_ids: Tensor[int] of category IDs [batch]
            - centers: Tensor[float] of crop centers used in preprocessing [batch, 2]
            - scales: Tensor[float] of scale factors used in preprocessing [batch, 2]
        """
        image_ids, category_ids, centers, scales = gt
        paf, heatmaps = output
        batch_size = heatmaps.shape[0]

        all_preds, all_maxvals, all_image_ids, all_category_ids = [], [], [], []

        for idx in range(batch_size):
            img_id = int(image_ids[idx])
            img_preds, img_scores = [], []

            # Compute the inverse affine transform used during preprocessing
            center = centers[idx]
            scale = scales[idx]
            rotate = 0
            input_size = [self.input_height, self.input_width]

            inv_trans = compute_affine_transform(
                center, scale, rotate, input_size, inv=True
            )
            inv_trans_tensor = torch.tensor(inv_trans, dtype=torch.float32)

            candidate, subset = getKeypointsFromPredictions(
                paf[idx].unsqueeze(0),
                heatmaps[idx].unsqueeze(0),
                self.input_height,
                self.input_width,
            )

            if len(subset) > 0:
                for person in subset:
                    person_preds = np.zeros((17, 2))
                    person_scores = np.zeros(17)

                    valid_coco_indices = []
                    keypoints_to_transform = []
                    for op_idx, coco_idx in enumerate(OPENPOSE_TO_COCO):
                        if op_idx == 1:  # Skip neck
                            continue
                        if op_idx < len(person) and person[op_idx] >= 0:
                            kp_idx = int(person[op_idx])
                            if kp_idx < len(candidate):
                                kp = candidate[kp_idx]
                                if kp[2] > self.min_score:
                                    # Map keypoint from cropped (input) space back to original image
                                    valid_coco_indices.append(coco_idx)
                                    keypoints_to_transform.append(kp[:2])
                                    person_scores[coco_idx] = kp[2]

                    if keypoints_to_transform:
                        kpts_tensor = torch.tensor(
                            keypoints_to_transform, dtype=torch.float32
                        )
                        transformed_kpts = apply_affine_to_coordinates(
                            kpts_tensor, inv_trans_tensor
                        ).numpy()
                        for i, coco_idx in enumerate(valid_coco_indices):
                            person_preds[coco_idx] = transformed_kpts[i]

                    img_preds.append(person_preds)
                    img_scores.append(person_scores)
                    all_image_ids.append(img_id)
                    all_category_ids.append(category_ids[idx])

            if not img_preds:
                img_preds.append(np.zeros((17, 2)))
                img_scores.append(np.zeros(17))
                all_image_ids.append(img_id)
                all_category_ids.append(category_ids[idx])

            all_preds.extend(img_preds)
            all_maxvals.extend(img_scores)

        preds = np.stack(all_preds)
        maxvals = np.stack(all_maxvals)
        self._store_predictions(preds, maxvals, all_image_ids, all_category_ids)
