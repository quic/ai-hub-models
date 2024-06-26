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

from qai_hub_models.models.mobilenet_v2.model import MobileNetV2
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
    convert_all_depthwise_to_per_tensor,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 5

# Weights downloaded from https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/torch_mobilenetv2_w8a8_state_dict.pth
QUANTIZED_WEIGHTS = "torch_mobilenetv2_w8a8_state_dict.pth"
DEFAULT_ENCODINGS = "mobilenet_v2_quantized_encodings.json"


class MobileNetV2Quantizable(AIMETQuantizableMixin, MobileNetV2):
    """MobileNetV2 with post train quantization support."""

    def __init__(
        self,
        quant_sim_model: QuantizationSimModel,
    ) -> None:
        # Input is already normalized by sim_model. Disable it in the wrapper model.
        MobileNetV2.__init__(self, quant_sim_model.model, normalize_input=False)
        AIMETQuantizableMixin.__init__(
            self,
            quant_sim_model,
        )

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> "MobileNetV2Quantizable":
        """
        Parameters:
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on imagenette.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        # Load Model
        model = MobileNetV2.from_pretrained()
        input_shape = cls.get_input_spec()["image_tensor"][0]
        # Following
        # https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/mobilenetv2/model/model_definition.py#L64
        model = prepare_model(model)
        equalize_model(model, input_shape)

        aimet_config = get_default_aimet_config()
        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=aimet_config,
            dummy_input=torch.rand(input_shape),
        )
        convert_all_depthwise_to_per_tensor(sim.model)
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        return cls(sim)
