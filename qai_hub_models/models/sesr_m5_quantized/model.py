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
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models._shared.sesr.common import _load_sesr_source_model
from qai_hub_models.models.sesr_m5.model import (
    NUM_CHANNELS,
    NUM_LBLOCKS,
    SCALING_FACTOR,
    SESR_M5,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/sesr/model/model_cards/sesr_m5_4x_w8a8.json:
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/sesr_m5_4x_checkpoint_int8.pth
# and
# https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.js
# Encodings were generated with AIMET QuantSim library
QUANTIZED_WEIGHTS = "sesr_m5_4x_checkpoint_int8.pth"
AIMET_ENCODINGS = "sesr_m5_quantized_encodings.json"
AIMET_CONFIG = "default_config_per_channel.json"


class SESR_M5Quantizable(AIMETQuantizableMixin, SESR_M5):
    """QuickSRNetLarge with post train quantization support.

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        sesr_model: QuantizationSimModel,
    ) -> None:
        SESR_M5.__init__(self, sesr_model.model)
        AIMETQuantizableMixin.__init__(
            self, sesr_model, needs_onnx_direct_aimet_export=False
        )

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> SESR_M5Quantizable:
        # Load Model
        sesr = _load_sesr_source_model(
            MODEL_ID, MODEL_ASSET_VERSION, SCALING_FACTOR, NUM_CHANNELS, NUM_LBLOCKS
        )
        input_shape = SESR_M5.get_input_spec()["image"][0]
        equalize_model(sesr, input_shape)

        # Download weights and quantization parameters
        weights = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, QUANTIZED_WEIGHTS
        ).fetch()
        aimet_config = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, AIMET_CONFIG
        ).fetch()

        # Load the model weights and quantization parameters
        state_dict = torch.load(weights, map_location=torch.device("cpu"))["state_dict"]
        # Here we collapse before loading the quantized weights.
        # The model is collapsed pre-quantization - see
        # https://github.com/quic/aimet-model-zoo/blob/d09d2b0404d10f71a7640a87e9d5e5257b028802/aimet_zoo_torch/common/super_resolution/models.py#L110
        sesr.collapse()
        sesr.load_state_dict(state_dict)
        sim = QuantizationSimModel(
            sesr,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=aimet_config,
            dummy_input=torch.rand(input_shape),
        )
        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, AIMET_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        sim.model.eval()

        return cls(sim)
