# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import numpy as np
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class ColorizationEvaluator(BaseEvaluator):
    """Evaluator for Computing Colorfulness of the predicted output."""

    def __init__(self):
        self.reset()

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        """
        Compute colorfulness of the predicted output.

        Args:
            output: torch.Tensor with shape (B, 2, 256, 256)
                predicted out in AB format (colors)
            gt: torch.Tensor with shape (B, 1, 256, 256)
                ground truth image in L format (lightness)
        """
        output, gt = output.numpy(), gt.numpy()
        for i in range(output.shape[0]):
            output_lab = np.concatenate((gt[i], output[i].transpose(1, 2, 0)), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

            output_img = (output_bgr * 255.0).round().astype(np.uint8)
            colorfulness = self.image_colorfulness(output_img)
            self.colorfulness.append(colorfulness)

    def image_colorfulness(self, image: np.ndarray) -> float:
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(image.astype(np.float32))
        B, G, R = np.array(B), np.array(G), np.array(R)
        # compute rg = R - G
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd**2) + (ybStd**2))
        meanRoot = np.sqrt((rbMean**2) + (ybMean**2))
        # derive the "colorfulness" metric
        colorfulness = stdRoot + (0.3 * meanRoot)
        return colorfulness

    def reset(self):
        self.colorfulness = []

    def get_accuracy_score(self) -> float:
        return np.mean(self.colorfulness)

    def formatted_accuracy(self) -> str:
        return f"Colorfulness: {self.get_accuracy_score():.3f}"
