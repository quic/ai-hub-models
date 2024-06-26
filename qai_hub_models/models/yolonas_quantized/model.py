# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet import (
    AIMETQuantizableMixin,
    constrain_quantized_inputs_to_image_range,
)

# isort: on

from typing import Optional

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models.yolonas.model import DEFAULT_WEIGHTS, YoloNAS
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import tie_observers

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_ENCODINGS = "yolonas_quantized_encodings.json"


class YoloNASQuantizable(AIMETQuantizableMixin, YoloNAS):
    """Exportable Quantized YoloNAS bounding box detector, end-to-end."""

    def __init__(
        self,
        sim_model: QuantizationSimModel,
    ) -> None:
        # Sim model will already include postprocessing
        torch.nn.Module.__init__(self)
        AIMETQuantizableMixin.__init__(self, sim_model)
        self.model = sim_model.model

    @classmethod
    def from_pretrained(
        cls,
        weights_name: Optional[str] = DEFAULT_WEIGHTS,
        aimet_encodings: str | None = "DEFAULT",
        include_postprocessing: bool = True,
    ) -> "YoloNASQuantizable":
        """Load YoloNAS from a weightfile created by the source YoloNAS repository."""
        fp16_model = YoloNAS.from_pretrained(
            weights_name,
            include_postprocessing=include_postprocessing,
        )
        fp16_model.class_dtype = torch.int8

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
        Run YoloNASQuantizable on `image`, and produce a
            predicted set of bounding boxes and associated class probabilities.

        See YoloNAS model for details.
        """
        return self.model(image)
