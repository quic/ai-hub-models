# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet import (
    AIMETQuantizableMixin,
    tie_observers,
    constrain_quantized_inputs_to_image_range,
)

# isort: on

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models.hrnet_pose.model import HRNetPose
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
DEFAULT_ENCODINGS = "hrnet_pose_quantized_encodings.json"


class HRNetPoseQuantizable(AIMETQuantizableMixin, HRNetPose):
    """HRNetPose with post training quantization support

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        hrnet_model: QuantizationSimModel,
    ) -> None:
        HRNetPose.__init__(self, hrnet_model.model)
        AIMETQuantizableMixin.__init__(self, hrnet_model)

    @classmethod
    def from_pretrained(
        cls, aimet_encodings: str | None = "DEFAULT"
    ) -> HRNetPoseQuantizable:
        model = HRNetPose.from_pretrained()
        input_shape = HRNetPose.get_input_spec()["image"][0]
        model = prepare_model(model)
        equalize_model(model, input_shape)

        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=get_default_aimet_config(),
            dummy_input=torch.rand(input_shape),
        )
        tie_observers(sim)
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        final_model = cls(sim)
        return final_model
