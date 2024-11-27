# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.yolov7.model import DEFAULT_WEIGHTS, YoloV7
from qai_hub_models.utils.quantization import HubQuantizableMixin

MODEL_ID = __name__.split(".")[-2]


class YoloV7Quantizable(HubQuantizableMixin, YoloV7):
    """Exportable quantizable YoloV7 bounding box detector."""

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ) -> YoloV7Quantizable:
        model = super().from_pretrained(
            ckpt_name,
            include_postprocessing=include_postprocessing,
        )
        model.class_dtype = torch.uint8
        return model

    def get_quantize_options(self) -> str:
        return "--range_scheme min_max"
