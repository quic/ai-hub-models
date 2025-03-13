# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.resnext50.model import ResNeXt50

MODEL_ID = __name__.split(".")[-2]


class ResNeXt50Quantizable(ResNeXt50):
    pass
