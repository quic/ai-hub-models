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

from qai_hub_models.models.deeplabv3_plus_mobilenet.model import DeepLabV3PlusMobilenet
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
    tie_observers,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4
DEFAULT_ENCODINGS = "deeplabv3_plus_mobilenet_quantized_encodings.json"


class DeepLabV3PlusMobilenetQuantizable(AIMETQuantizableMixin, DeepLabV3PlusMobilenet):
    """
    DeepLabV3PlusMobileNet with post train quantization support.

    Supports only 8 bit weights and activations
    """

    def __init__(
        self,
        deeplabv3_model: QuantizationSimModel,
    ) -> None:
        DeepLabV3PlusMobilenet.__init__(
            self, deeplabv3_model.model, normalize_input=False
        )
        AIMETQuantizableMixin.__init__(self, deeplabv3_model)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
        normalize_input: bool = True,
    ) -> "DeepLabV3PlusMobilenetQuantizable":
        # Load Model
        fp16_model = DeepLabV3PlusMobilenet.from_pretrained(
            normalize_input=normalize_input
        )
        input_shape = cls.get_input_spec()["image"][0]

        model = prepare_model(fp16_model)
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
