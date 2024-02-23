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

import os

import torch
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from qai_hub.client import DatasetEntries

from qai_hub_models.models._shared.ffnet.model import FFNet
from qai_hub_models.utils.base_model import SourceModelFormat, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
FFNET_AIMET_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "aimet_config.json")
)


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

    def get_hub_compile_options(
        self, target_runtime: TargetRuntime, other_compile_options: str = ""
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, other_compile_options
        )
        return compile_options + " --quantize_full_type int8 --quantize_io"

    @classmethod
    def default_aimet_encodings(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        variant_name: str,
        aimet_encodings: str | None = "DEFAULT",
    ) -> "FFNetQuantizable":
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
        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = cls.default_aimet_encodings()
            load_encodings_to_sim(sim, aimet_encodings)

        sim.model.eval()
        return cls(sim)

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        return SourceModelFormat.ONNX

    def get_calibration_data(
        self, target_runtime: TargetRuntime, input_spec: InputSpec | None = None
    ) -> DatasetEntries | None:
        # Do not provide calibration data
        return None
