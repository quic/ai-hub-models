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
from aimet_torch.v1.quantsim import QuantizationSimModel

from qai_hub_models.models._shared.face_detection.model import FaceDetLite_model
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = "face_det_lite_quantized"
MODEL_ASSET_VERSION = 2
DEFAULT_ENCODINGS = "face_det_lite_quantized_encodings.json"  # TODO neesd to be updated for foot tracknet.


class FaceDetLiteQuantizable(AIMETQuantizableMixin, FaceDetLite_model):
    """FaceDetLite_model with post train quantization support.

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        model: QuantizationSimModel,
    ) -> None:
        FaceDetLite_model.__init__(self, model.model)
        AIMETQuantizableMixin.__init__(
            self,
            model,
        )

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> FaceDetLiteQuantizable:
        """
        Parameters:
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on imagenette.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        model = FaceDetLite_model.from_pretrained()
        input_shape = cls.get_input_spec()["input"][0]
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

        final_model = cls(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()

            final_model.load_encodings(
                str(aimet_encodings),
                strict=False,
                partial=False,
                requires_grad=None,
                allow_overwrite=None,
            )

        return final_model
