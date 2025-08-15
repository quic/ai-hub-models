# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata


class ValentiniEvaluator(BaseEvaluator):
    """Evaluator for speech enhancement on Valentini dataset."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
    ):
        """
        Args:
            target_sample_rate: Sample rate to resample audio to (Hz)
        """
        self.target_sample_rate = target_sample_rate
        self.reset()

    def add_batch(
        self,
        output: Collection[torch.Tensor],
        target: Collection[torch.Tensor],
    ):
        """
        Args:
            output: Enhanced audio waveforms [batch, samples]
            target: Clean reference waveforms [batch, samples]
        """

        for enhanced, clean in zip(output, target):
            enhanced = enhanced.squeeze()
            clean = clean.squeeze()

            self.enhanced.append(enhanced.cpu().numpy())
            self.clean.append(clean.cpu().numpy())

    def reset(self):
        """Reset stored predictions and targets"""
        self.enhanced = []
        self.clean = []

    def _compute_metrics(self) -> tuple[float, float]:
        """Compute PESQ and STOI metrics"""
        pesq_scores = []
        stoi_scores = []

        for enhanced, clean in zip(self.enhanced, self.clean):
            pesq_i, stoi_i = self._compute_single_metrics(
                clean, enhanced, self.target_sample_rate
            )

            pesq_scores.append(pesq_i)
            stoi_scores.append(stoi_i)

        avg_pesq = sum(pesq_scores) / len(pesq_scores) if pesq_scores else 0
        avg_stoi = sum(stoi_scores) / len(stoi_scores) if stoi_scores else 0
        return avg_pesq, avg_stoi

    @staticmethod
    def _compute_single_metrics(
        clean: np.ndarray,
        enhanced: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, float]:
        """Compute metrics for a single audio pair"""
        pesq_score = 0
        stoi_score = 0

        pesq_score = pesq(sample_rate, clean, enhanced, "wb")  # 'wb' = wideband PESQ
        stoi_score = stoi(clean, enhanced, sample_rate, extended=False)

        return pesq_score, stoi_score

    def get_accuracy_score(self) -> float:
        pesq, _ = self._compute_metrics()
        return pesq

    def formatted_accuracy(self) -> str:
        pesq, stoi = self._compute_metrics()
        return f"PESQ: {pesq:.3f}, STOI: {stoi:.3f}"

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Perceptual Evaluation of Speech Quality",
            unit="PESQ",
            description="A measure of quality degradation between original and predicted quality.",
        )
