# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Collection

import numpy as np
import torch
from numpy.linalg import norm

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class FaceAttribNetEvaluator(BaseEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(self):
        self.id_features = {}
        self.id_total = 0
        self.feature_total = 0

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]):
        """
        gt should be a tuple of tensors with the following tensors:
            - image_ids of shape (batch_size,)
            - image heights of shape (batch_size,)
            - image widths of shape (batch_size,)

        output should be a tuple of tensors with the following tensors:
            - identity feature map with shape (batch_size, 512)
        """
        id_features, _, _, _, _, _ = output
        image_ids, image_idxs, _, _ = gt

        """
        extract id and index information from filename, for example image filename
        ***_27_0, extracted id is 27, and image index for id 27 is 0. Normally there
        are multiple images with same id, index with 0 for enroll, other index for query
        """
        for i in range(len(image_ids)):
            id_str = str(image_ids[i].item())
            id_idx = str(image_idxs[i].item())
            if id_str not in self.id_features:
                self.id_features[id_str] = {}
                self.id_total += 1
            if id_idx not in self.id_features[id_str]:
                self.feature_total += 1
            self.id_features[id_str][id_idx] = np.squeeze(
                id_features[i].detach().numpy()
            )

    def reset(self):
        self.id_features = {}
        self.id_total = 0
        self.feature_total = 0

    def get_accuracy_score(self):
        """
        query_feas is 10 x 512 array, enroll_feas is 5 x 512 array.
        cos_sim is 10 x 5 array for similarity score.
        return mean value of cos_sim.
        """
        query_feas = []
        enroll_feas = []
        mask_sign = np.full((self.feature_total - self.id_total, self.id_total), -1)
        mask_offset = np.full((self.feature_total - self.id_total, self.id_total), 1)
        query_pos = 0
        enroll_pos = 0
        for key_0, value_0 in self.id_features.items():
            for key_1, value_1 in value_0.items():
                if key_1 == "0":
                    enroll_feas.append(value_1)
                else:
                    query_feas.append(value_1)
                    mask_sign[query_pos][enroll_pos] = 1
                    mask_offset[query_pos][enroll_pos] = 0
                    query_pos += 1
            enroll_pos += 1
        query_feas_total = np.array(query_feas)
        enroll_feas_total = np.array(enroll_feas)
        cos_sim = (
            np.matmul(query_feas_total, enroll_feas_total.T)
            / norm(query_feas_total, axis=1)[:, None]
            / norm(enroll_feas_total, axis=1)[None, :]
        )
        cos_sim[cos_sim < 0] = 0
        cos_sim = cos_sim * mask_sign + mask_offset

        return np.mean(cos_sim)

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} mAP"
