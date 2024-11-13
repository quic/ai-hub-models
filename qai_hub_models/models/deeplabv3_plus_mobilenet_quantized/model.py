# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.deeplabv3_plus_mobilenet.model import DeepLabV3PlusMobilenet
from qai_hub_models.utils.quantization import HubQuantizableMixin

MODEL_ID = __name__.split(".")[-2]


class DeepLabV3PlusMobilenetQuantizable(HubQuantizableMixin, DeepLabV3PlusMobilenet):
    pass
