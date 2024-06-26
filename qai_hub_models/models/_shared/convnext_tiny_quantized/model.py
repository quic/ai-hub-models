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

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from torchvision.models.convnext import LayerNorm2d as ConvNextLayerNorm2d
from torchvision.ops.misc import Permute

from qai_hub_models.models._shared.common import replace_module_recursively
from qai_hub_models.models.convnext_tiny.model import DEFAULT_WEIGHTS, ConvNextTiny
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
)


# The ConvNext LayerNorm uses a functional LayerNorm that is currently not
# automatically handled by AIMET (AIMET-3928). With this fix, the LayerNorms
# will not get quantization observers.
class AIMETLayerNorm2d(nn.Sequential):
    def __init__(self, orig_layer_norm: ConvNextLayerNorm2d):
        layer_norm = nn.LayerNorm(
            orig_layer_norm.normalized_shape,
            eps=orig_layer_norm.eps,
            elementwise_affine=orig_layer_norm.elementwise_affine,
        )
        layer_norm.bias = orig_layer_norm.bias
        layer_norm.weight = orig_layer_norm.weight
        super().__init__(
            Permute([0, 2, 3, 1]),
            layer_norm,
            Permute([0, 3, 1, 2]),
        )


class ConvNextTinyQuantizableBase(AIMETQuantizableMixin, ConvNextTiny):
    def __init__(
        self,
        quant_sim_model: QuantizationSimModel,
    ) -> None:
        # Input is already normalized by sim_model. Disable it in the wrapper model.
        ConvNextTiny.__init__(self, quant_sim_model.model, normalize_input=False)
        AIMETQuantizableMixin.__init__(
            self,
            quant_sim_model,
        )

    @classmethod
    @abstractmethod
    def _default_aimet_encodings(cls) -> str | Path:
        """
        Default AIMET encodings path.
        """
        ...

    @classmethod
    @abstractmethod
    def _output_bw(cls) -> int:
        """
        Quantization bitwidth of activations.
        """
        ...

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_WEIGHTS,
        aimet_encodings: str | None = "DEFAULT",
    ) -> "ConvNextTinyQuantizableBase":
        """
        Parameters:
          weights:
            Weights of the model. See Torchvision ConvNext for information of
            the format of this object.
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on imagenette.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        # Load Model
        model = ConvNextTiny.from_pretrained(weights=weights)

        replace_module_recursively(
            model,
            ConvNextLayerNorm2d,
            AIMETLayerNorm2d,
        )

        input_shape = cls.get_input_spec()["image_tensor"][0]
        model = prepare_model(model)
        equalize_model(model, input_shape)

        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=cls._output_bw(),
            config_file=get_default_aimet_config(),
            dummy_input=torch.rand(input_shape),
        )
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = cls._default_aimet_encodings()
            load_encodings_to_sim(sim, aimet_encodings)

        return cls(sim)
