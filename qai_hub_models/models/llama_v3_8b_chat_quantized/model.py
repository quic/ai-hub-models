# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Llama3Base_Quantized,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4
DEFAULT_ENCODINGS = "llama3.encodings"
DEFAULT_ENCODINGS_ZIP = DEFAULT_ENCODINGS + ".zip"

NUM_LAYERS = 32
NUM_SPLITS = 5
NUM_LAYERS_PER_SPLIT = 9

# Hugging face repo name and url
HF_REPO_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MIN_MEMORY_RECOMMENDED = 80


class Llama3_Quantized(Llama3Base_Quantized):
    def __init__(self, huggingface_model_name: str = HF_REPO_NAME, *args, **kwargs):
        super().__init__(
            huggingface_model_name=huggingface_model_name,  # type: ignore[misc]
            min_memory_recommended=MIN_MEMORY_RECOMMENDED,
            *args,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        aimet_encodings: str | None = "DEFAULT",
        huggingface_model_name: str = HF_REPO_NAME,
    ) -> Llama3_Quantized:
        """
        Load a pre-trained Llama 3 (8B) model from Meta via HuggingFace.

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
        huggingface_model_name:
            Name or URL of the HuggingFace model. Change this if you want to
            change the weights.
        """
        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = os.path.join(
                    CachedWebModelAsset.from_asset_store(
                        MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS_ZIP
                    ).fetch(extract=True),
                    DEFAULT_ENCODINGS,
                )

        return cls(
            aimet_encodings=aimet_encodings,
            sequence_length=sequence_length,
            context_length=context_length,
            huggingface_model_name=huggingface_model_name,
        )

    @staticmethod
    def get_output_names(num_hidden_layers: int = NUM_LAYERS):
        return Llama3Base_Quantized.get_output_names(
            num_hidden_layers=num_hidden_layers
        )

    @staticmethod
    def get_input_spec(
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        return Llama3Base_Quantized._get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=4096,
            num_key_value_heads=8,
            num_attention_heads=32,
        )
