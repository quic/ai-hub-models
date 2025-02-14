# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub.client import QuantizeDtype

from qai_hub_models.models.convnext_tiny.model import ConvNextTiny
from qai_hub_models.utils.quantization import HubQuantizableMixin

MODEL_ID = __name__.split(".")[-2]


class ConvNextTinyW8A16Quantizable(HubQuantizableMixin, ConvNextTiny):
    @staticmethod
    def get_activations_dtype() -> QuantizeDtype:
        return QuantizeDtype.INT16
