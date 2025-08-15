# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.evaluators.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.utils.image_processing import (
    apply_affine_to_coordinates,
    compute_affine_transform,
)


class MovenetPoseEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for MoveNet pose estimation models"""

    def __init__(self, height, width, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_height = height
        self.input_width = width

    def add_batch(
        self, output: torch.Tensor | tuple[torch.Tensor], gt: list[torch.Tensor]
    ):
        """
        Processes MoveNet outputs and converts them to COCO-format keypoint predictions.

        Args:
            output: Model predictions, shape [batch, N_people, 17, 3] (x, y, confidence)
            gt: list with the following Tensors:
                - image_ids: Tensor[int] of image IDs [batch]
                - category_ids: Tensor[int] of category IDs [batch]
                - centers: Tensor[float] of bounding box centers [batch, 2]
                - scale: Tensor[float] of scale factors [batch, 2]
        """
        if isinstance(output, tuple):
            output = output[0]

        image_ids, category_ids, centers, scales = gt
        batch_size = output.shape[0]

        for idx in range(batch_size):
            img_id = int(image_ids[idx])
            cat_id = int(category_ids[idx])
            center = centers[idx].cpu().numpy()
            scale = scales[idx].cpu().numpy()
            input_size = [self.input_height, self.input_width]

            # Get inverse affine transform
            inv_trans = compute_affine_transform(center, scale, 0, input_size, inv=True)
            inv_trans_tensor = torch.tensor(inv_trans, dtype=torch.float32)

            # Loop over all detected people
            n_people = output.shape[1]
            for person_idx in range(n_people):
                kp_with_scores = output[idx, person_idx].cpu().numpy()
                keypoints = kp_with_scores[:, :2]
                scores = kp_with_scores[:, 2]

                # Scale normalized (0-1) keypoints to input size
                keypoints *= input_size

                # Flip (x, y)
                keypoints = np.flip(keypoints, axis=1).copy()

                # Apply inverse affine transform
                keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
                coords_tf = apply_affine_to_coordinates(
                    keypoints_tensor, inv_trans_tensor
                ).numpy()

                # Store predictions
                self._store_predictions(
                    np.expand_dims(coords_tf, 0),  # (1, 17, 2)
                    np.expand_dims(scores, 0),  # (1, 17)
                    torch.tensor([img_id]),
                    torch.tensor([cat_id]),
                )
