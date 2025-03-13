# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.resnet_3d.model import ResNet3D

MODEL_ID = __name__.split(".")[-2]


class ResNet3DQuantizable(ResNet3D):
    pass
