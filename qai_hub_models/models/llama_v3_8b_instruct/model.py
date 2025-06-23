# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
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
    sample_input,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_CHECKPOINT = "llama3_ckpt"
DEFAULT_CHECKPOINT_ZIP = DEFAULT_CHECKPOINT + ".zip"

NUM_LAYERS = 32
NUM_SPLITS = 5
NUM_LAYERS_PER_SPLIT = 9
HIDDEN_SIZE = 4096
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging face repo name and url
HF_REPO_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MIN_MEMORY_RECOMMENDED = 80


class Llama3_8B(Llama3Base):
    def __init__(
        self, checkpoint: str | os.PathLike | Path = HF_REPO_NAME, *args, **kwargs
    ):
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            min_memory_recommended=MIN_MEMORY_RECOMMENDED,
            *args,
            **kwargs,
        )
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
        _skip_optimizations: list[str] | None = None,
    ) -> Llama3_8B:
        """
        Load a pre-trained Llama 3 (8B) model from Meta via HuggingFace.

        checkpoint:
            Local path or Hugging Face model name of floating point checkpoint.
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
        aimet_encodings:
            Path to AIMET quantization encodings file.
        """
        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            _skip_optimizations=_skip_optimizations,
        )

    @staticmethod
    def get_output_names(num_hidden_layers: int = NUM_LAYERS):
        return Llama3Base.get_output_names(num_hidden_layers=num_hidden_layers)

    @staticmethod
    def get_input_spec(
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        return Llama3Base._get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=HIDDEN_SIZE,
            num_key_value_heads=NUM_KEY_VALUE_HEADS,
            num_attention_heads=NUM_ATTN_HEADS,
        )


class Llama3_8B_AIMETOnnx(Llama3Base_AIMETOnnx):
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
    ) -> Llama3_8B_AIMETOnnx:

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

    @staticmethod
    def get_output_names(num_hidden_layers: int = NUM_LAYERS):
        return Llama3Base_AIMETOnnx.get_output_names(num_hidden_layers)

    @staticmethod
    def get_input_spec(
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        return Llama3Base._get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=HIDDEN_SIZE,
            num_key_value_heads=NUM_KEY_VALUE_HEADS,
            num_attention_heads=NUM_ATTN_HEADS,
        )

    def _sample_inputs_impl(self, input_spec: InputSpec | None = None):
        if input_spec is None:
            input_spec = self.input_specs
        return sample_input(
            input_spec,
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
        )
