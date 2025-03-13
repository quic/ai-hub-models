# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.wideresnet50.model import WideResNet50

MODEL_ID = __name__.split(".")[-2]


class WideResNet50Quantizable(WideResNet50):
    pass
