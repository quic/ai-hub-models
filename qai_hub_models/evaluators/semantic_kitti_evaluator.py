# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata


class SemanticKittiEvaluator(BaseEvaluator):
    """Evaluator for comparing Semantic segmentation output against ground truth."""

    def __init__(self, n_classes: int, learning_map: dict, learning_ignore: dict):
        self.n_classes = n_classes
        self.learning_map = learning_map
        self.include = []
        self.ignore = []
        for key, value in learning_ignore.items():
            if value:
                self.ignore.append(key)
            else:
                self.include.append(key)
        self.reset()

    def reset(self):
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes)).long()
        self.ones = None
        self.last_scan_size = None

    def add_batch(
        self, output: torch.Tensor, gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Args:
            output (torch.Tensor): Model predictions.
            gt_data (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                p_x (torch.tensor): x coordinates of lidar points with shape [max_points,]
                p_y (torch.tensor): y coordinates of lidar points with shape [max_points,]
                label (torch.tensor): semantic labels with shape [max_points,]
        """
        p_x, p_y, label = gt
        unproj_argmax_list = []
        for i in range(output.shape[0]):
            proj_argmax = output[i].argmax(dim=0)
            unproj_argmax_list.append(
                proj_argmax[p_y[i : i + 1], p_x[i : i + 1]].to(torch.int32)
            )
        unproj_argmax = torch.concat(unproj_argmax_list)

        x_row = unproj_argmax.reshape(-1)  # de-batchify
        y_row = label.reshape(-1)  # de-batchify

        temp = []
        for i in y_row:
            if i == -1:
                temp.append(self.learning_map[0])
            else:
                temp.append(self.learning_map[int(i)])

        y_row = torch.tensor(temp)

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones(idxs.shape[-1]).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True
        )

    def getIoU(self) -> float:
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp

        intersection = tp
        union = tp + fp + fn + 1e-15
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean

    def get_accuracy_score(self) -> float:
        return self.getIoU()

    def formatted_accuracy(self) -> str:
        return f"{self.getIoU():.3f} mIOU"

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Mean Intersection Over Union",
            unit="mIOU",
            description="Overlap of predicted and expected segmentation divided by the union size.",
        )
