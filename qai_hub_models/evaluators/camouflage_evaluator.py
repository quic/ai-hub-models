# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from py_sod_metrics import MAE, Emeasure, Smeasure, WeightedFmeasure

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.bgnet.app import postprocess_masks


class CamouflageEvaluator(BaseEvaluator):
    """Evaluator for comparing segmentation output against ground truth"""

    def __init__(self, metrics=("mae", "smeasure", "wfmeasure", "emeasure")):
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.sm = Smeasure()
        self.wfm = WeightedFmeasure()
        self.em = Emeasure()
        self.mae = MAE()
        self.results: dict = {m: [] for m in self.metrics}

    def add_batch(self, pred_images: torch.Tensor, gt_images: torch.Tensor):
        """
        Process a batch of segmentation predictions and ground truth masks.
        Args:
            pred_images (torch.Tensor):  output predictions with shape
                [batch_size, 1, height, width]
            gt_images (torch.Tensor): Ground truth masks with shape
                [batch_size, 1, height, width]
        """
        if isinstance(pred_images, tuple):
            pred_images = pred_images[0]

        pred_np = postprocess_masks(pred_images, gt_images.shape[-2:])
        gt_np = gt_images.cpu().numpy().astype(np.uint8)

        for pred, gt in zip(pred_np, gt_np):
            self.sm.step(pred=pred, gt=gt, normalize=True)
            self.wfm.step(pred=pred, gt=gt, normalize=True)
            self.em.step(pred=pred, gt=gt, normalize=True)
            self.mae.step(pred=pred, gt=gt, normalize=True)

    def smeasure(self) -> float:
        """Returns the S-measure (structural similarity) score
        Higher values indicate better segmentation quality.
        """
        return float(self.sm.get_results()["sm"])

    def wfmeasure(self) -> float:
        """Returns the weighted F-measure score
        Higher values indicate better boundary accuracy.
        """
        return float(self.wfm.get_results()["wfm"])

    def mae_acc(self) -> float:
        """Returns the Mean Absolute Error
        Lower values indicate better pixel-wise accuracy.
        """
        return float(self.mae.get_results()["mae"])

    def emasure(self) -> float:
        """Returns the mean E-measure (enhanced alignment measure) score
        Higher values indicate better region and boundary alignment.
        """
        return float(self.em.get_results()["em"]["curve"].mean())

    def get_accuracy_score(self) -> float:
        return self.smeasure()

    def formatted_accuracy(self) -> str:
        parts = [
            f"MAE: {self.mae_acc():.4f}",
            f"Smeasure: {self.smeasure():.4f}",
            f"wFmeasure: {self.wfmeasure():.4f}",
            f"Emeasure: {self.emasure():.4f}",
        ]
        return ", ".join(parts)
