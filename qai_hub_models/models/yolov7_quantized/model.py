# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.common import Precision
from qai_hub_models.models.yolov7.model import DEFAULT_WEIGHTS, YoloV7

MODEL_ID = __name__.split(".")[-2]


class YoloV7Quantizable(YoloV7):
    """Exportable quantizable YoloV7 bounding box detector."""

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> YoloV7Quantizable:
        model = super().from_pretrained(
            weights_name,
            include_postprocessing=include_postprocessing,
            split_output=split_output,
        )
        model.class_dtype = torch.uint8
        return model

    def get_hub_quantize_options(self, precision: Precision) -> str:
        return "--range_scheme min_max"
