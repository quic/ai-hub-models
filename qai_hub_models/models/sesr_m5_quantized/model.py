# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.sesr_m5.model import SESR_M5

MODEL_ID = __name__.split(".")[-2]


class SESR_M5Quantizable(SESR_M5):
    pass
