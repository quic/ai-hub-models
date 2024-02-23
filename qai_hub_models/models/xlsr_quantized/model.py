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
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models.xlsr.model import XLSR, _load_xlsr_source_model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/xlsr/model/model_cards/xlsr_4x_w8a8.json:
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_february_artifacts/xlsr_4x_checkpoint_int8.pth
# and
# https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.js
# Encodings were generated with AIMET QuantSim library
XLSR_QUANTIZED_WEIGHTS = "xlsr_4x_checkpoint_int8.pth"
AIMET_ENCODINGS = "aimet_quantization_encodings.json"
AIMET_CONFIG = "default_config_per_channel.json"
SCALING_FACTOR = 4


class XLSRQuantizable(AIMETQuantizableMixin, XLSR):
    """XLSR with post training quantization suport

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        xlsr_model: QuantizationSimModel,
    ) -> None:
        XLSR.__init__(self, xlsr_model.model)
        AIMETQuantizableMixin.__init__(
            self, xlsr_model, needs_onnx_direct_aimet_export=True
        )

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> XLSRQuantizable:
        """
        Parameters:
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on BSD300.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        xlsr = _load_xlsr_source_model()
        input_shape = XLSR.get_input_spec()["image"][0]

        weights = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, XLSR_QUANTIZED_WEIGHTS
        ).fetch()
        aimet_config = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, AIMET_CONFIG
        ).fetch()

        # Load the model weights and quantization parameters
        state_dict = torch.load(weights, map_location=torch.device("cpu"))["state_dict"]
        xlsr.load_state_dict(state_dict)
        sim = QuantizationSimModel(
            xlsr,
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

        return cls(sim)
