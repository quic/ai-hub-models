# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models import Precision
from qai_hub_models.models.templates.detr.model import DETR

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class DETRResNet50(DETR):
    DEFAULT_WEIGHTS = "facebook/detr-resnet-50"

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model=model)

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
