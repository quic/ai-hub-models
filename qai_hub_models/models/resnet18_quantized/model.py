# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.resnet18.model import ResNet18

MODEL_ID = __name__.split(".")[-2]


class ResNet18Quantizable(ResNet18):
    pass
