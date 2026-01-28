# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from transformers import ConditionalDetrForObjectDetection

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models._shared.detr.model import DETR

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "microsoft/conditional-detr-resnet-50"
MODEL_ASSET_VERSION = 1


class ConditionalDETRResNet50(DETR):
    """Exportable DETR model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        return cls(ConditionalDetrForObjectDetection.from_pretrained(ckpt_name))

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(image_height, image_width, score_threshold=0.4)
