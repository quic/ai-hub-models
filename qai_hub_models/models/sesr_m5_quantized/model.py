# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet import (
    AIMETQuantizableMixin,
)

# isort: on

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models._shared.super_resolution.model import DEFAULT_SCALE_FACTOR
from qai_hub_models.models.sesr_m5.model import SESR_M5
from qai_hub_models.utils.aimet.config_loader import get_default_per_tensor_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 5
DEFAULT_ENCODINGS = "sesr_m5_quantized_encodings.json"


class SESR_M5Quantizable(AIMETQuantizableMixin, SESR_M5):
    """SESR_M5 with post train quantization support.

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        sesr_model: QuantizationSimModel,
        scale_factor: int,
    ) -> None:
        SESR_M5.__init__(self, sesr_model.model, scale_factor)
        AIMETQuantizableMixin.__init__(self, sesr_model)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
        scale_factor: int = DEFAULT_SCALE_FACTOR,
    ) -> SESR_M5Quantizable:
        # Load Model
        sesr = SESR_M5.from_pretrained(scale_factor)
        input_shape = SESR_M5.get_input_spec()["image"][0]
        sesr = prepare_model(sesr)
        equalize_model(sesr, input_shape)

        sim = QuantizationSimModel(
            sesr,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=get_default_per_tensor_aimet_config(),
            dummy_input=torch.rand(input_shape),
        )
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        return cls(sim, scale_factor)
