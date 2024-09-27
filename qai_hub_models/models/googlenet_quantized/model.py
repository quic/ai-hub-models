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

from typing import Optional

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from qai_hub import Device

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.googlenet.model import GoogLeNet
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
    tie_observers,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4
DEFAULT_ENCODINGS = "googlenet_quantized_encodings.json"


class GoogLeNetQuantizable(AIMETQuantizableMixin, GoogLeNet):
    """GoogleNet with post train quantization support.

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        sim_model: QuantizationSimModel,
    ) -> None:
        # Input is already normalized by sim_model. Disable it in the wrapper model.
        GoogLeNet.__init__(self, sim_model.model, normalize_input=False)
        AIMETQuantizableMixin.__init__(
            self,
            sim_model,
            needs_onnx_direct_aimet_export=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> "GoogLeNetQuantizable":
        """
        Parameters:
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on imagenette.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        model = GoogLeNet.from_pretrained()
        input_shape = cls.get_input_spec()["image_tensor"][0]

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

        return cls(sim)

    # TODO(12424) remove this once encodings export correctly
    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, other_compile_options, device
        )
        if target_runtime not in [
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ]:
            compile_options += " --quantize_full_type int8"
        return compile_options
