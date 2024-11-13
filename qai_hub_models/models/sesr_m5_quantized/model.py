# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.sesr_m5.model import SESR_M5
from qai_hub_models.utils.quantization import HubQuantizableMixin

MODEL_ID = __name__.split(".")[-2]


class SESR_M5Quantizable(HubQuantizableMixin, SESR_M5):
    pass
