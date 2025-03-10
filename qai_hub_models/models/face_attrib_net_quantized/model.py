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
    tie_observers,
)

# isort: on

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models.face_attrib_net.model import FaceAttribNet
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_ENCODINGS = "face_attrib_net_quantized_encodings.json"

OUT_NAMES = [
    "id_feature",
    "liveness_feature",
    "eye_closeness",
    "glasses",
    "mask",
    "sunglasses",
]


class FaceAttribNetQuantizable(AIMETQuantizableMixin, FaceAttribNet):
    """FaceAttribNet with post train quantization support.

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        fanet_model: QuantizationSimModel,
    ) -> None:
        # Input is already normalized by sim_model. Disable it in the wrapper model.
        FaceAttribNet.__init__(self, fanet_model.model)
        AIMETQuantizableMixin.__init__(
            self,
            fanet_model,
        )

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> FaceAttribNetQuantizable:
        """
        Parameters:
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on imagenette.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        model = FaceAttribNet.from_pretrained()
        input_shape = cls.get_input_spec()["image"][0]

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
