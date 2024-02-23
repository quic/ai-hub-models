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

from qai_hub_models.models.hrnet_pose.model import HRNetPose
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/hrnet_posenet/models/model_cards/hrnet_posenet_w8a8.json:
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_march_artifacts/hrnet_posenet_W8A8_state_dict.pth
# Encodings were generated with AIMET QuantSim export
QUANTIZED_WEIGHTS = "hrnet_posenet_W8A8_state_dict.pth"
AIMET_ENCODINGS = "hrnetpose_aimet_quantization_encodings.json"
AIMET_CONFIG = "default_config_per_channel.json"


class HRNetPoseQuantizable(AIMETQuantizableMixin, HRNetPose):
    """HRNetPose with post training quantization suport

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        hrnet_model: QuantizationSimModel,
    ) -> None:
        HRNetPose.__init__(self, hrnet_model.model)
        AIMETQuantizableMixin.__init__(
            self, hrnet_model, needs_onnx_direct_aimet_export=True
        )

    @classmethod
    def from_pretrained(cls) -> HRNetPoseQuantizable:
        model = HRNetPose.from_pretrained()
        input_shape = HRNetPose.get_input_spec()["image"][0]
        equalize_model(model, input_shape)

        weights = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, QUANTIZED_WEIGHTS
        ).fetch()
        aimet_config = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, AIMET_CONFIG
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, AIMET_ENCODINGS
        ).fetch()

        # Load the model weights and quantization parameters
        state_dict = torch.load(weights, map_location=torch.device("cpu"))
        new_state_dict = {"model." + key: value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=aimet_config,
            dummy_input=torch.rand(input_shape),
        )
        load_encodings_to_sim(sim, aimet_encodings)

        return cls(sim)
