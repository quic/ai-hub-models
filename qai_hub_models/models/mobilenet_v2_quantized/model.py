# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet import (
    AIMETQuantizableMixin,
    HubCompileOptionsInt8Mixin,
)

# isort: on

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models.mobilenet_v2.model import (
    MobileNetV2,
    _load_mobilenet_v2_source_model,
)
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import SourceModelFormat, TargetRuntime

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

# Weights downloaded from https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/torch_mobilenetv2_w8a8_state_dict.pth
QUANTIZED_WEIGHTS = "torch_mobilenetv2_w8a8_state_dict.pth"
DEFAULT_ENCODINGS = "encodings.json"


class MobileNetV2Quantizable(
    HubCompileOptionsInt8Mixin, AIMETQuantizableMixin, MobileNetV2
):
    """MobileNetV2 with post train quantization support."""

    def __init__(
        self,
        quant_sim_model: QuantizationSimModel,
    ) -> None:
        MobileNetV2.__init__(self, quant_sim_model.model)
        AIMETQuantizableMixin.__init__(
            self,
            quant_sim_model,
        )

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        return SourceModelFormat.ONNX

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
        model_fp32 = _load_mobilenet_v2_source_model(
            keep_sys_path=True,
        )
        input_shape = MobileNetV2(None).get_input_spec()["image_tensor"][0]
        # Following
        # https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/mobilenetv2/model/model_definition.py#L64
        equalize_model(model_fp32, input_shape)

        # Download weights and quantization parameters
        weights = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, QUANTIZED_WEIGHTS
        ).fetch()
        aimet_config = get_default_aimet_config()

        # Load the QAT/PTQ tuned model_fp32 weights
        checkpoint = torch.load(weights, map_location=torch.device("cpu"))
        state_dict = {
            k.replace("classifier.1", "classifier"): v
            for k, v in checkpoint["state_dict"].items()
        }
        model_fp32.load_state_dict(state_dict)
        sim = QuantizationSimModel(
            model_fp32,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=aimet_config,
            dummy_input=torch.rand(input_shape),
        )

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        sim.model.eval()
        return cls(sim)
