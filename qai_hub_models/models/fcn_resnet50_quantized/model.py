# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.fcn_resnet50.model import FCN_ResNet50

MODEL_ID = __name__.split(".")[-2]


class FCN_ResNet50Quantizable(FCN_ResNet50):
    pass
