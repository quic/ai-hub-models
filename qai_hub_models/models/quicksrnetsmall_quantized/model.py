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

from qai_hub_models.models.common import SourceModelFormat, TargetRuntime
from qai_hub_models.models.quicksrnetsmall.model import QuickSRNetSmall
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config_legacy_v2
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/quicksrnet/model/model_cards/quicksrnet_small_4x_w8a8.json:
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/quicksrnet_small_4x_checkpoint_int8.pth
# and
# https://raw.githubusercontent.com/quic/aimet/release-aimet-1.23/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.js
# Encodings were generated with AIMET QuantSim library
QUANTIZED_WEIGHTS = "quicksrnet_small_4x_checkpoint_int8.pth"
AIMET_ENCODINGS = "aimet_quantization_encodings.json"
SCALING_FACTOR = 4


class QuickSRNetSmallQuantizable(AIMETQuantizableMixin, QuickSRNetSmall):
    """QuickSRNetSmall with post train quantization support.
    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        quicksrnet_model: QuantizationSimModel,
    ) -> None:
        QuickSRNetSmall.__init__(self, quicksrnet_model.model)
        AIMETQuantizableMixin.__init__(
            self, quicksrnet_model, needs_onnx_direct_aimet_export=True
        )

    @classmethod
    def from_pretrained(
        cls, aimet_encodings: str | None = "DEFAULT"
    ) -> "QuickSRNetSmallQuantizable":
        """
        Parameters:
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on BSD300.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        # Load Model
        quicksrnet = QuickSRNetSmall.from_pretrained()
        input_shape = quicksrnet.get_input_spec()["image"][0]
        equalize_model(quicksrnet, input_shape)

        # Download weights and quantization parameters
        weights = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, QUANTIZED_WEIGHTS
        ).fetch()
        aimet_config = get_default_aimet_config_legacy_v2()

        # Load the model weights and quantization parameters
        # In this particular instance, the state_dict keys from the model are all named "model.<expected name>"
        # where <expected name> is the name of each key in the weights file - without the word model.
        # We rename all the keys to add the word model
        state_dict = torch.load(weights, map_location=torch.device("cpu"))["state_dict"]
        new_state_dict = {"model." + key: value for key, value in state_dict.items()}
        quicksrnet.load_state_dict(new_state_dict)
        sim = QuantizationSimModel(
            quicksrnet,
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

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        if target_runtime == TargetRuntime.QNN:
            return SourceModelFormat.ONNX
        else:
            return SourceModelFormat.TORCHSCRIPT
