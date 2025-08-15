# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata
from qai_hub_models.models.facemap_3dmm.utils import (
    project_landmark,
    transform_landmark_coordinates,
)


class FaceMap3DMMEvaluator(BaseEvaluator):
    """Evaluator for keypoint-based Face landmark estimation with private dataset using Normalized Mean Error."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
    ):
        self.reset()
        self.image_height = image_height
        self.image_width = image_width

    def reset(self):
        """Resets the collected predictions."""
        self.results = []

    def add_batch(self, output: torch.Tensor, gt: list[torch.Tensor]):
        """
        gt should be a list of tensors with the following tensors:
            - image_ids of shape (batch_size,)
            - gt_landmarks of shape (batch_size, 68, 2)
                - The ground truth landmarks, where each landmark is represented by its x and y coordinates.
            - image widths of shape (batch_size, 4)
                - The face box in the images, represented as a tensor with shape [4] and layout [left, right, top, bottom].
        output should be a tensor predicted from face landmark model
            - with shape (batch_size, 265)
        """
        image_ids, gt_landmarks, bboxes = gt

        for i in range(len(image_ids)):
            gt_landmark = gt_landmarks[i].numpy()
            pred_landmark = project_landmark(output[i])
            transform_landmark_coordinates(
                pred_landmark, bboxes[i], self.image_height, self.image_width
            )

            pred_landmark = pred_landmark[:, :2].numpy()
            eye_length = np.sqrt(np.sum((gt_landmark[36, :] - gt_landmark[45, :]) ** 2))
            gt_landmark = np.concatenate((gt_landmark[17:30, :], gt_landmark[36:48, :]))
            pred_landmark = np.concatenate(
                (pred_landmark[17:30, :], pred_landmark[36:48, :])
            )

            error = np.sqrt(np.sum(np.power(pred_landmark - gt_landmark, 2), 1))
            nme = np.mean(error) / eye_length
            self.results.append(nme)

    def get_mean_nme(self):
        self.results = [nme for nme in self.results if nme < 0.1]
        return np.mean(self.results)

    def get_accuracy_score(self):
        return self.get_mean_nme()

    def formatted_accuracy(self) -> str:
        mean_nme = self.get_accuracy_score()
        return f"Mean NME: {mean_nme:.4f}"

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Normalized Mean Error",
            unit="NME",
            description="Average distance between predicted and expected landmark, weighted by the typical scale for each landmark.",
        )
