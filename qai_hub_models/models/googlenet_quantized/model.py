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

from qai_hub_models.models.googlenet.model import GoogLeNet
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
    tie_aimet_observer_groups,
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
        cls._tie_pre_concat_quantizers(sim)
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        sim.model.eval()
        return cls(sim)

    @classmethod
    def _tie_pre_concat_quantizers(cls, sim: QuantizationSimModel):
        """
        This ties together the output quantizers prior to concatenations. This
        prevents unnecessary re-quantization during the concatenation.
        """
        blocks = [
            sim.model.net.inception3a,
            sim.model.net.inception3b,
            sim.model.net.inception4a,
            sim.model.net.inception4b,
            sim.model.net.inception4c,
            sim.model.net.inception4d,
            sim.model.net.inception4e,
            sim.model.net.inception5a,
            sim.model.net.inception5b,
        ]

        idx = 3
        groups = []
        for block in blocks:
            groups.append(
                [
                    getattr(block.branch1, f"module_relu_{idx}"),
                    getattr(getattr(block.branch2, "1"), f"module_relu_{idx+2}"),
                    getattr(getattr(block.branch3, "1"), f"module_relu_{idx+4}"),
                    getattr(getattr(block.branch4, "1"), f"module_relu_{idx+5}"),
                ]
            )
            idx += 6

        tie_aimet_observer_groups(groups)
