# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.mobilenet_v2.model import MobileNetV2
from qai_hub_models.utils.quantization import HubQuantizableMixin

MODEL_ID = __name__.split(".")[-2]


class MobileNetV2Quantizable(HubQuantizableMixin, MobileNetV2):
    pass
