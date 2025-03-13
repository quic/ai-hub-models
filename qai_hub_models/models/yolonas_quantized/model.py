# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.common import Precision
from qai_hub_models.models.yolonas.model import DEFAULT_WEIGHTS, YoloNAS

MODEL_ID = __name__.split(".")[-2]


class YoloNASQuantizable(YoloNAS):
    """Exportable quantizable YoloNAS bounding box detector."""

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ) -> YoloNAS:
        model = super().from_pretrained(
            ckpt_name,
            include_postprocessing=include_postprocessing,
        )
        model.class_dtype = torch.uint8
        return model

    def get_hub_quantize_options(self, precision: Precision) -> str:
        return "--range_scheme min_max"
