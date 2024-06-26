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

import os
from typing import Type, TypeVar

import torch
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models._shared.ffnet.model import FFNet

MODEL_ID = __name__.split(".")[-2]
FFNET_AIMET_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "aimet_config.json")
)

FFNetQuantizableType = TypeVar("FFNetQuantizableType", bound="FFNetQuantizable")


class FFNetQuantizable(AIMETQuantizableMixin, FFNet):
    """
    FFNet with post train quantization support.

    Supports only 8-bit weights and activations.
    """

    def __init__(
        self,
        ffnet_model: FFNet,
    ) -> None:
        FFNet.__init__(self, ffnet_model.model)
        AIMETQuantizableMixin.__init__(self, ffnet_model)

    @classmethod
    def default_aimet_encodings(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls: Type[FFNetQuantizableType],
        variant_name: str,
        aimet_encodings: str | None = "DEFAULT",
    ) -> FFNetQuantizableType:
        ffnet = FFNet.from_pretrained(variant_name).model

        input_shape = FFNetQuantizable.get_input_spec()["image"][0]

        fold_all_batch_norms(ffnet, [input_shape])

        ffnet = prepare_model(ffnet)

        sim = QuantizationSimModel(
            ffnet,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=FFNET_AIMET_CONFIG,
            dummy_input=torch.rand(input_shape),
        )
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = cls.default_aimet_encodings()
            load_encodings_to_sim(sim, aimet_encodings)

        return cls(sim)
