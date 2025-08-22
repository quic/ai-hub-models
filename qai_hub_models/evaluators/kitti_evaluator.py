# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.datasets.kitti import KittiDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.utils.kitti import eval_class
from qai_hub_models.models.centernet.util import ddd_post_process


class KittiEvaluator(BaseEvaluator):
    """Evaluator for comparing Semantic segmentation output against ground truth."""

    def __init__(self, decode, max_dets=100, peak_thresh=0.2):
        self.decode = decode
        self.max_dets = max_dets
        self.peak_thresh = peak_thresh
        self.data_path = KittiDataset().data_path
        self.reset()

    def reset(self):
        self.dt_annos = []
        self.gt_annos = []

    def add_batch(
        self,
        output: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """
        output : tuple of following tensor
            hm (torch.Tensor): Heatmap with the shape of
                [B, num_classes, H//4, W//4].
            dep (torch.Tensor): depth value with the
                shape of [B, 1, H//4, W//4].
            rot (torch.Tensor): Rotation value with the
                shape of [B, 8, H//4, W//4].
            dim (torch.Tensor): Size value with the shape
                of [B, 3, H//4, W//4].
            wh (torch.Tensor): Width/Height value with the
                shape of [B, 2, H//4, W//4].
            reg (torch.Tensor): 2D regression value with the
                shape of [B, 2, H//4, W//4].
        gt : tuple of following tensor
            img_id (torch.Tensor): Image id tensor with the
                shape of [B,].
            c (torch.Tensor): Center of bbox with the shape of
                [B, 2].
            s (torch.Tensor): Scale of bbox with the shape of
                [B, 2].
            calib (torch.Tensor): Calibration matrix with the
                shape of [B, 3, 4].
        """
        hm, dep, rot, dim, wh, reg = output
        img_id, c, s, calib = gt
        dets = self.decode(hm, rot, dep, dim, wh=wh, reg=reg, K=self.max_dets)
        dets = dets.detach().numpy()
        dets = ddd_post_process(
            dets,
            list(np.array(c)),
            list(np.array(s)),
            hm.shape[2:],
            list(np.array(calib)),
        )

        for i in range(len(dets)):
            results = dets[i]
            # filter only cars class based on score threshold for evaluation
            results = results[results[:, -1] == 1]
            results = results[results[:, -2] > self.peak_thresh]

            if results.shape[0] != 0:
                # pred annotations
                k = list(results)
                dt_annotations = {}
                dt_annotations["name"] = np.array(["Car" for _ in k])
                dt_annotations["truncated"] = np.array([0.0 for _ in k])
                dt_annotations["occluded"] = np.array([0 for _ in k])
                dt_annotations["alpha"] = np.array([x[0] for x in k])
                dt_annotations["bbox"] = np.array(
                    [[float(info) for info in x[1:5]] for x in k]
                ).reshape(-1, 4)
                # dimensions will convert hwl format to standard lhw(camera) format.
                dt_annotations["dimensions"] = np.array(
                    [[float(info) for info in x[5:8]] for x in k]
                ).reshape(-1, 3)[:, [2, 0, 1]]
                dt_annotations["location"] = np.array(
                    [[float(info) for info in x[8:11]] for x in k]
                ).reshape(-1, 3)
                dt_annotations["rotation_y"] = np.array(
                    [float(x[11]) for x in k]
                ).reshape(-1)
                dt_annotations["score"] = np.array([float(x[12]) for x in k])
                self.dt_annos.append(dt_annotations)

                # gt annotations
                label_path = self.data_path / f"label_2/{int(img_id[i]):06d}.txt"
                with open(label_path) as f:
                    lines = f.readlines()
                content = [line.strip().split(" ") for line in lines]
                gt_annotations = {}
                gt_annotations["name"] = np.array([x[0] for x in content])
                gt_annotations["truncated"] = np.array([float(x[1]) for x in content])
                gt_annotations["occluded"] = np.array([int(x[2]) for x in content])
                gt_annotations["alpha"] = np.array([float(x[3]) for x in content])
                gt_annotations["bbox"] = np.array(
                    [[float(info) for info in x[4:8]] for x in content]
                ).reshape(-1, 4)
                # dimensions will convert hwl format to standard lhw(camera) format.
                gt_annotations["dimensions"] = np.array(
                    [[float(info) for info in x[8:11]] for x in content]
                ).reshape(-1, 3)[:, [2, 0, 1]]
                gt_annotations["location"] = np.array(
                    [[float(info) for info in x[11:14]] for x in content]
                ).reshape(-1, 3)
                gt_annotations["rotation_y"] = np.array(
                    [float(x[14]) for x in content]
                ).reshape(-1)
                if len(content) != 0 and len(content[0]) == 16:  # have score
                    gt_annotations["score"] = np.array([float(x[15]) for x in content])
                else:
                    gt_annotations["score"] = np.zeros([len(gt_annotations["bbox"])])
                self.gt_annos.append(gt_annotations)

    def get_accuracy_score(self) -> float:
        num_parts = len(self.dt_annos) // 100 + 1
        bbox, _, _ = eval_class(
            self.gt_annos,
            self.dt_annos,
            difficultys=[0],
            num_parts=num_parts,
        )
        return float(bbox)

    def formatted_accuracy(self) -> str:
        num_parts = len(self.dt_annos) // 100 + 1
        bbox, aos, bev = eval_class(
            self.gt_annos,
            self.dt_annos,
            difficultys=[0, 1, 2],
            num_parts=num_parts,
        )

        bbox_str = ", ".join([f"{v:.2f}" for v in bbox])
        bev_str = ", ".join([f"{v:.2f}" for v in bev])
        aos_str = ", ".join([f"{v:.2f}" for v in aos])

        return f"{bbox_str} AP-(E,M,H) {aos_str} AOS-(E,M,H) {bev_str} BEV-(E,M,H)"
