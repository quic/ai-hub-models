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

from qai_hub_models.models.fcn_resnet50.model import FCN_ResNet50
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
    tie_observers,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_ENCODINGS = "fcn_resnet50_quantized_encodings.json"


class FCN_ResNet50Quantizable(AIMETQuantizableMixin, FCN_ResNet50):
    """
    FCN_ResNet50 with post train quantization support.

    Supports only 8 bit weights and activations
    """

    def __init__(
        self,
        model: QuantizationSimModel,
    ) -> None:
        FCN_ResNet50.__init__(self, model.model)
        AIMETQuantizableMixin.__init__(self, model)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> "FCN_ResNet50Quantizable":
        # Load Model
        fp16_model = FCN_ResNet50.from_pretrained()
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

    def forward(self, image: torch.Tensor):
        """
        Run FCN_ResNet50Quantizable on `image`, and produce a segmentation mask.

        See FCN_ResNet50 model for details.
        """
        return self.model(image)
