# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import onnx
import torch

from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Llama3Base,
    Llama3Base_AIMETOnnx,
)
from qai_hub_models.models._shared.llm._utils import _set_lm_head_to_8b
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

AIMET_ONNX_INSTALLED = False
try:
    import aimet_common.quantsim as qs
    from aimet_common.defs import QuantizationDataType, QuantScheme
    from aimet_onnx import quantsim
    from aimet_onnx.quantsim import QuantizationSimModel

    AIMET_ONNX_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    print(
        "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Install qai-hub-models on a Linux machine to use quantized models."
    )

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_CHECKPOINT = "llama32_ckpt"
DEFAULT_CHECKPOINT_ZIP = DEFAULT_CHECKPOINT + ".zip"
NUM_LAYERS = 16
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 8
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging face repo name and url
HF_REPO_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MIN_MEMORY_RECOMMENDED = 30


class Llama3_2_1B(Llama3Base):
    min_memory_recommended = MIN_MEMORY_RECOMMENDED

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        *args,
        **kwargs,
    ):
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            *args,
            **kwargs,
        )

    def _verify_ckpt(self):
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == NUM_LAYERS
            and self.llm_config.hidden_size == HIDDEN_SIZE
            and self.llm_config.num_attention_heads == NUM_ATTN_HEADS
            and self.llm_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        load_pretrained: bool = True,
        _skip_optimizations: list[str] | None = None,
    ) -> Llama3_2_1B:
        """
        Load a pre-trained Llama 3.2 (1B) model from Meta via HuggingFace.

        checkpoint:
            Local path or Hugging Face name of floating point checkpoint.
        sequence_length:
            Instantiate with this token sequence length input. A longer
            sequence length means the model is capable of processing more
            tokens at once. This can only be set to greater than one to process
            prompts, since responses are auto-regressive in nature and require
            this to be 1.
        context_length:
            Total context length of model. Longer context length means the
            model is more capable of making longer connections in the input
            prompt. However, it also hurts runtime performance (both time-to-
            first-token and tokens-per-second), so this is a tradeoff that may
            depend on the use case.
        """
        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            load_pretrained=load_pretrained,
            _skip_optimizations=_skip_optimizations,
        )

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        return Llama3Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
        )


class Llama3_2_1B_AIMETOnnx(Llama3Base_AIMETOnnx):
    def __init__(self, checkpoint: str | os.PathLike | Path | None, *args, **kwargs):
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            *args,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path | None = "DEFAULT",
        host_device: torch.device | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        fp_model: torch.nn.Module | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> Llama3_2_1B_AIMETOnnx:
        """
        Load weight from Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Args:

        - checkpoint: Path to previously calibrated AIMET encodings and ONNX
          models. Note that encodings are sensitive to AIMET ONNX versions.
          If passing None, initializes without encodings.
        """
        if checkpoint == "DEFAULT":
            checkpoint = os.path.join(
                CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_CHECKPOINT_ZIP
                ).fetch(extract=True),
                DEFAULT_CHECKPOINT,
            )

            # Generate necessary ONNX models
            if fp_model is not None:
                cls.create_onnx_models(
                    checkpoint=checkpoint,
                    fp_model=fp_model,
                    context_length=context_length,
                    export_sequence_lengths=[sequence_length],
                    host_device=host_device,
                )

                cls.save_tokenizer_and_config(checkpoint=checkpoint, fp_model=fp_model)

        return super().from_pretrained(
            checkpoint=checkpoint,
            host_device=host_device,
            sequence_length=sequence_length,
            context_length=context_length,
            fp_model=fp_model,
            _skip_quantsim_creation=_skip_quantsim_creation,
        )

    @classmethod
    def create_quantsim(
        cls, onnx_model: onnx.ModelProto, host_device: torch.device
    ) -> QuantizationSimModel:
        """
        onnx_model: ONNX Model to create QuantSim model.
        host_device: Device that the QuantSim model must be placed on.
        """
        if not AIMET_ONNX_INSTALLED:
            raise ImportError(
                "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
                "Install qai-hub-models on a Linux machine to use quantized models."
            )
        default_config = get_aimet_config_path("default_config_llama")
        # Tie Quantizers for Concat Op
        quantsim.op_types_to_tie_qtzrs = ["Concat"]
        quantsim._tie_qtzrs = True
        # Ignore Slice and Constant outputs
        quantsim.op_outputs_to_ignore.append("Slice")
        quantsim.op_outputs_to_ignore.append("Constant")
        qs.encoding_version = "1.0.0"

        quant_sim = QuantizationSimModel(
            model=onnx_model,
            param_type="int4",
            activation_type="int16",
            quant_scheme=QuantScheme.min_max,
            config_file=default_config,
            providers=cls.get_ort_providers(host_device),
        )

        # Setting the LM head weights to 8-bit.
        _set_lm_head_to_8b(quant_sim)

        # Set all activation quantizers to float16
        for op_name, qc_op in quant_sim.qc_quantize_op_dict.items():
            if op_name in quant_sim.activation_names:
                qc_op.reset_encoding_stats()
                qc_op.data_type = QuantizationDataType.float
                qc_op.bitwidth = 16

        return quant_sim

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        return Llama3Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
        )
