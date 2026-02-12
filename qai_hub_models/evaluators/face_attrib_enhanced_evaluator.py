# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from numpy.linalg import norm

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata


class FaceAttribNetEnhancedEvaluator(BaseEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(self) -> None:
        # dict[person ID, dict[image ID, Feature Embedding]]
        self.id_features: dict[int, dict[int, np.ndarray]] = {}
        # total number of persons
        self.id_total = 0
        # total number of images
        self.feature_total = 0

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor],
        gt: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Adds a batch of model outputs and corresponding ground truth data.

        Parameters
        ----------
        output
            Output from the `face_attrib_net_enhanced` model, including:

            prob
                Tensor (float) with range [0, 1], shape (N, M) where N is batch_size
                and M is number of attributes (5).
            feature_embed
                Tensor (float) with shape (N, 512).
        gt
            Ground truth labels from the `FaceAttribEnhancedDataset`, corresponding to the batch:

            person_ids
                Tensor (int) of shape [N], each is the ID of an individual in this image.
            image_ids
                Tensor (int) of shape [N], each is the ID of the image.
        """
        _, feature_embed = output
        person_names, image_ids = gt

        """
        extract id and index information from filename, for example image filename
        ***_Burke_7415, extracted id is Burke, and image index for id Burke is 7415. Normally there
        are multiple images with same id. Take the first one for enroll, other index for query
        """
        for i in range(len(person_names)):
            id_person = int(person_names[i].item())
            id_idx = int(image_ids[i].item())
            if id_person not in self.id_features:
                self.id_features[id_person] = {}
                self.id_total += 1
            if id_idx not in self.id_features[id_person]:
                self.feature_total += 1
            self.id_features[id_person][id_idx] = np.squeeze(
                feature_embed[i].detach().numpy()
            )

    def reset(self) -> None:
        """Reset model evaluation result variables."""
        self.id_features = {}
        self.id_total = 0
        self.feature_total = 0

    def get_accuracy_score(self) -> float:
        """
        Calculates and returns the accuracy score for model evaluation.

        The accuracy is computed as the average of cosine similarity scores between pairs of face embeddings.
        Each embedding is a 512-dimensional vector representing facial features.

        For each image pair:
        - If both images are of the same person, the raw cosine similarity is used.
        - If the images are of different people, the similarity is calculated as (1 - cosine similarity).

        This approach ensures that high similarity scores for same-person pairs and low scores for different-person pairs
        both contribute positively to the overall accuracy.

        The final accuracy score is the mean of these adjusted similarity values across all image pairs.

        Returns
        -------
        accuracy_metric : float
            Accuracy metric in range [0, 1].

        """
        # query_feas is 80 x 512 array, enroll_feas is 20 x 512 array.
        # cos_sim is 80 x 20 array for similarity score.
        # return mean value of cos_sim.
        query_feas = []
        enroll_feas = []
        mask_sign = np.full((self.feature_total - self.id_total, self.id_total), -1)
        mask_offset = np.full((self.feature_total - self.id_total, self.id_total), 1)
        query_pos = 0
        for enroll_pos, feas_by_idx in enumerate(self.id_features.values()):
            for i, fea in enumerate(feas_by_idx.values()):
                if i == 0:
                    enroll_feas.append(fea)
                else:
                    query_feas.append(fea)
                    mask_sign[query_pos][enroll_pos] = 1
                    mask_offset[query_pos][enroll_pos] = 0
                    query_pos += 1
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
        """
        Return accuracy in formatted string

        Returns
        -------
        formatted_accuracy_string : str
            formatted string of accuracy report
        """
        return f"{self.get_accuracy_score():.3f} Cosine Similarity"

    def get_metric_metadata(self) -> MetricMetadata:
        """
        Return accuracy in MetricMetadata

        Returns
        -------
        MetricMetadata
        """
        return MetricMetadata(
            name="Cosine Similarity",
            unit="",
            description="Similarity between the predicted facial features and the expected.",
            range=(0.0, 1.0),
            float_vs_device_threshold=0.1,
        )
