# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import torch

from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Llama3Base,
    Llama3Base_AIMETOnnx,
)
from qai_hub_models.models._shared.llm.model import determine_precision_from_checkpoint
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

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


DEFAULT_PRECISION = Precision.w4
SUPPORTED_PRECISIONS = [Precision.w4, Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4: "llama32_ckpt_w4",
    Precision.w4a16: "llama32_ckpt_w4a16",
}


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
        precision: Precision = DEFAULT_PRECISION,
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
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            precision = determine_precision_from_checkpoint(checkpoint) or precision
            if precision not in SUPPORTED_PRECISIONS:
                available_precisions = [str(p) for p in SUPPORTED_PRECISIONS]
                raise ValueError(
                    f"This model is not supported for {str(precision)} precision. "
                    f"Models are available in following precisions: {','.join(available_precisions)}."
                )
            if precision not in DEFAULT_CHECKPOINT:
                available_checkpoints = [str(p) for p in DEFAULT_CHECKPOINT]
                raise ValueError(
                    f"No checkpoint is available for this model in {str(precision)} precision. If you would "
                    f"like to continue with this precision, please generate a local quantized checkpoint. "
                    f"Checkpoints are available in the following precisions: {','.join(available_checkpoints)}."
                )
            precision_checkpoint = DEFAULT_CHECKPOINT[precision]
            checkpoint = os.path.join(
                CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, precision_checkpoint + ".zip"
                ).fetch(extract=True),
                precision_checkpoint,
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
            precision=precision,
            fp_model=fp_model,
            _skip_quantsim_creation=_skip_quantsim_creation,
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
