# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.onnx.helpers import (
    safe_torch_onnx_export,
)
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
    set_aimet_log_level,
)

# isort: on

import logging

import aimet_onnx
import onnx
from aimet_common.defs import QuantScheme
from aimet_onnx.cross_layer_equalization import equalize_model
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    load_encodings_to_sim,
    op_outputs_to_ignore,
)
from onnxsim import simplify
from qai_hub.client import Device
from transformers import WhisperConfig

from qai_hub_models.models._shared.hf_whisper.model import (
    HfWhisperDecoder,
    HfWhisperEncoder,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.base_model import Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs

# Define the path to the AIMET configuration file
WHISPER_AIMET_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "qsim_config_per_tensor.json")
)


# Using AIMETOnnxQuantizableMixin.forward, ignoring conflict with HfWhisperEncoder
class WhisperEncoderQuantizableBase(AIMETOnnxQuantizableMixin, HfWhisperEncoder):  # type: ignore[misc]
    """A class that represents a quantizable Whisper encoder."""

    def __init__(self, config: WhisperConfig, sim_model: QuantizationSimModel) -> None:
        """
        Initializes the WhisperEncoderQuantizableBase class.

        Parameters
        ----------
            sim_model (QuantizationSimModel): The quantization simulation model.
        """
        HfWhisperEncoder.__init__(self, config)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)

    @classmethod
    def make_torch_model(cls) -> HfWhisperEncoder:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | os.PathLike | None = "DEFAULT",
    ) -> WhisperEncoderQuantizableBase:
        """
        Creates a WhisperEncoderQuantizableBase instance from a pre-trained model.

        Parameters
        ----------
            aimet_encodings (str | None, optional): The AIMET encodings. Defaults to "DEFAULT".

        Returns
        -------
            WhisperEncoderQuantizableBase: The created instance.
        """
        fp_encoder = cls.make_torch_model()
        aimet_encodings_path = aimet_encodings
        if aimet_encodings == "DEFAULT":
            # Load the ONNX model with AIMET encodings
            onnx_file, encodings_path = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load_model(onnx_file)
            aimet_encodings_path = encodings_path
        else:
            # Create a temporary ONNX model from the pre-trained encoder
            specs = fp_encoder._get_input_spec_for_instance()
            dummy_input = tuple(make_torch_inputs(specs))
            # Create a temporary directory for the ONNX model
            with qaihm_temp_dir() as tempdir:
                temp_model_path = os.path.join(tempdir, "model.onnx")
                # Export the decoder model to ONNX
                safe_torch_onnx_export(
                    fp_encoder,
                    dummy_input,
                    temp_model_path,
                    export_params=True,
                    input_names=list(specs.keys()),
                    output_names=fp_encoder._get_output_names_for_instance(),
                )
                # Simplify the ONNX model
                onnx_model, _ = simplify(temp_model_path)
                # Apply cross-layer equalization to the ONNX model
                equalize_model(onnx_model)

        if aimet_encodings == "DEFAULT":
            # Remove certain ops from the ignore list
            # TODO(#14827): Move this to aimet config after op_outputs_to_ignore is removed in future aimet
            [
                op_outputs_to_ignore.remove(op)
                for op in ["Transpose", "Split", "Reshape", "Gather"]
                if op in op_outputs_to_ignore
            ]
        # Create a QuantizationSimModel instance
        quant_sim = QuantizationSimModel(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=aimet_onnx.int8,
            activation_type=aimet_onnx.int16,
            config_file=WHISPER_AIMET_CONFIG,
        )

        if aimet_encodings_path is not None:
            # Load the AIMET encodings into the QuantizationSimModel instance
            with set_aimet_log_level(logging.WARN):
                load_encodings_to_sim(
                    quant_sim, str(aimet_encodings_path), strict=False
                )

        return cls(fp_encoder.config, quant_sim)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        """Override same func in AIMETOnnxQuantizableMixin."""
        return super(HfWhisperEncoder, self)._sample_inputs_impl(input_spec)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        return super(HfWhisperEncoder, self).get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )


# Using AIMETOnnxQuantizableMixin.forward, ignoring conflict with HfWhisperDecoder
class WhisperDecoderQuantizableBase(AIMETOnnxQuantizableMixin, HfWhisperDecoder):  # type: ignore[misc]
    """A class that represents a quantizable Whisper decoder."""

    def __init__(self, config: WhisperConfig, sim_model: QuantizationSimModel) -> None:
        """
        Initializes the WhisperDecoderQuantizableBase class.

        Parameters
        ----------
            sim_model (QuantizationSimModel): The quantization simulation model.
        """
        HfWhisperDecoder.__init__(self, config)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)

    @classmethod
    def make_torch_model(cls) -> HfWhisperDecoder:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | os.PathLike | None = "DEFAULT",
    ) -> WhisperDecoderQuantizableBase:
        """
        Creates a WhisperDecoderQuantizableBase instance from a pre-trained model.

        Parameters
        ----------
            aimet_encodings (str | None, optional): The AIMET encodings. Defaults to "DEFAULT".

        Returns
        -------
            WhisperDecoderQuantizableBase: The created instance.
        """
        fp_decoder = cls.make_torch_model()
        aimet_encodings_path = aimet_encodings
        if aimet_encodings == "DEFAULT":
            # Load the ONNX model with AIMET encodings
            onnx_file, encodings_path = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load(onnx_file)
            aimet_encodings_path = encodings_path
        else:
            # Create a temporary ONNX model from the pre-trained decoder
            specs = fp_decoder._get_input_spec_for_instance()
            dummy_input = tuple(make_torch_inputs(specs))
            # Create a temporary directory for the ONNX model
            with qaihm_temp_dir() as tempdir:
                temp_model_path = os.path.join(tempdir, "model.onnx")
                # Export the decoder model to ONNX
                safe_torch_onnx_export(
                    fp_decoder,
                    dummy_input,
                    temp_model_path,
                    export_params=True,
                    input_names=list(specs.keys()),
                    output_names=fp_decoder._get_output_names_for_instance(),
                )
                # Simplify the ONNX model
                onnx_model, _ = simplify(temp_model_path)
                # Apply cross-layer equalization to the ONNX model
                equalize_model(onnx_model)

        if aimet_encodings == "DEFAULT":
            # Remove certain ops from the ignore list
            # TODO(#14827): Move this to aimet config after op_outputs_to_ignore is removed in future aimet
            [
                op_outputs_to_ignore.remove(op)
                for op in ["Transpose", "Split", "Reshape", "Gather"]
                if op in op_outputs_to_ignore
            ]
        # Create a QuantizationSimModel instance
        quant_sim = QuantizationSimModel(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=aimet_onnx.int8,
            activation_type=aimet_onnx.int16,
            config_file=WHISPER_AIMET_CONFIG,
        )

        if aimet_encodings_path is not None:
            # Load the AIMET encodings into the QuantizationSimModel instance
            with set_aimet_log_level(logging.WARN):
                load_encodings_to_sim(
                    quant_sim, str(aimet_encodings_path), strict=False
                )

        return cls(fp_decoder.config, quant_sim)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        """Override same impl func in AIMETOnnxQuantizableMixin."""
        return super(HfWhisperDecoder, self)._sample_inputs_impl(input_spec)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        return super(HfWhisperDecoder, self).get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
