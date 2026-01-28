# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import cast

import torch
from qai_hub import Device
from transformers import MarianMTModel, MarianTokenizer
from typing_extensions import Self

from qai_hub_models.models._shared.opus_mt.model_adaptation import (
    QcMarianDecoder,
    QcMarianEncoder,
    apply_model_adaptations,
)
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "opus_mt_shared"
MODEL_ASSET_VERSION = 1

# Model configuration constants
MAX_SEQ_LEN_ENC = 256
MAX_SEQ_LEN_DEC = 256


class OpusMTEncoder(BaseModel):
    """
    OpusMT Encoder optimized for export and inference.

    It takes text input (input_ids) and produces cross attention
    key-value cache for the decoder.
    """

    def __init__(self, model: QcMarianEncoder) -> None:
        super().__init__()
        self.encoder = model

    @classmethod
    def from_pretrained(
        cls, opus_mt_version: str = "Helsinki-NLP/opus-mt-en-es"
    ) -> Self:
        return cls(OpusMT.get_opus_model(opus_mt_version)[0])

    def forward(
        self, input_ids: torch.Tensor, encoder_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            Input token IDs of shape (batch_size, sequence_length)
        encoder_attention_mask
            Attention mask of shape (batch_size, sequence_length)

        Returns
        -------
        tuple[torch.Tensor, ...]
            Cross attention key and value cache tensors
        """
        return self.encoder(input_ids, encoder_attention_mask)

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "input_ids": ((1, MAX_SEQ_LEN_ENC), "int32"),
            "encoder_attention_mask": ((1, MAX_SEQ_LEN_ENC), "int32"),
        }

    @staticmethod
    def get_output_names(num_layers: int = 6) -> list[str]:
        """Returns the output names for the encoder."""
        output_names = []
        for layer_idx in range(num_layers):
            output_names.append(f"block_{layer_idx}_cross_key_states")
            output_names.append(f"block_{layer_idx}_cross_value_states")
        return output_names

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if (
            precision == Precision.float
            and target_runtime.qairt_version_changes_compilation
        ):
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class OpusMTDecoder(BaseModel):
    """
    OpusMT Decoder optimized for export and inference.

    Wraps MarianDecoderMod to facilitate export.
    """

    def __init__(self, model: QcMarianDecoder) -> None:
        super().__init__()
        self.decoder = model
        self.num_layers = 6  # OpusMT has 6 decoder layers

    @classmethod
    def from_pretrained(
        cls, opus_mt_version: str = "Helsinki-NLP/opus-mt-en-es"
    ) -> Self:
        return cls(OpusMT.get_opus_model(opus_mt_version)[1])

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        position: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            Input token IDs of shape (batch_size, 1)
        encoder_attention_mask
            Encoder attention mask of shape (batch_size, encoder_seq_len)
        position
            Current position index of shape (1,)
        *past_key_values
            Past key-value states for self and cross attention

        Returns
        -------
        tuple[torch.Tensor, ...]
            Logits and updated key-value states
        """
        return self.decoder(
            input_ids, encoder_attention_mask, position, *past_key_values
        )

    @staticmethod
    def get_input_spec(
        num_layers: int = 6,
        attention_dim: int = 512,
        num_heads: int = 8,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        head_dim = attention_dim // num_heads

        specs = {
            "input_ids": ((1, 1), "int32"),
            "encoder_attention_mask": ((1, MAX_SEQ_LEN_ENC), "int32"),
            "position": ((1,), "int32"),
        }

        # Add past key-value states for each layer
        # Using transpose_key=False format (consistent with original notebook)
        for i in range(num_layers):
            specs[f"block_{i}_past_self_key_states"] = (
                (1, num_heads, MAX_SEQ_LEN_DEC - 1, head_dim),
                "float32",
            )
            specs[f"block_{i}_past_self_value_states"] = (
                (1, num_heads, MAX_SEQ_LEN_DEC - 1, head_dim),
                "float32",
            )
            specs[f"block_{i}_cross_key_states"] = (
                (1, num_heads, MAX_SEQ_LEN_ENC, head_dim),
                "float32",
            )
            specs[f"block_{i}_cross_value_states"] = (
                (1, num_heads, MAX_SEQ_LEN_ENC, head_dim),
                "float32",
            )

        return specs

    @staticmethod
    def get_output_names(num_layers: int = 6) -> list[str]:
        """Returns the output names for the decoder."""
        output_names = ["logits"]
        for layer_idx in range(num_layers):
            output_names.append(f"block_{layer_idx}_present_self_key_states")
            output_names.append(f"block_{layer_idx}_present_self_value_states")
        return output_names

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if (
            precision == Precision.float
            and target_runtime.qairt_version_changes_compilation
        ):
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class OpusMT(CollectionModel):
    """
    Base OpusMT translation model.

    This model consists of an encoder and decoder that work together
    to translate text between languages.
    """

    def __init__(
        self,
        encoder: OpusMTEncoder,
        decoder: OpusMTDecoder,
        hf_source: str,
    ) -> None:
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.hf_source = hf_source

    @classmethod
    @abstractmethod
    def get_opus_mt_version(cls) -> str:
        """Return the HuggingFace model identifier for this OpusMT variant."""

    @classmethod
    def get_opus_model(
        cls, opus_mt_version: str | None = None
    ) -> tuple[QcMarianEncoder, QcMarianDecoder]:
        # Load the pretrained model - this downloads pytorch_model.bin and other files
        orig_model = cast(
            MarianMTModel,
            MarianMTModel.from_pretrained(opus_mt_version or cls.get_opus_mt_version()),
        )
        orig_model.eval()

        # Apply model adaptations to optimize for QNN inference
        return apply_model_adaptations(orig_model)

    @classmethod
    def from_pretrained(cls) -> Self:
        """
        Load OpusMT model from pretrained weights.

        This will download the model files (including pytorch_model.bin) from HuggingFace Hub
        if they are not already cached locally.
        """
        # Load the original Marian model
        encoder_qc, decoder_qc = cls.get_opus_model()
        opus_mt_version = cls.get_opus_mt_version()

        # Wrap in our model classes
        encoder = OpusMTEncoder(encoder_qc)
        decoder = OpusMTDecoder(decoder_qc)
        return cls(encoder, decoder, opus_mt_version)


def get_tokenizer(hf_model_name: str) -> MarianTokenizer:
    """Get the tokenizer for the specified OpusMT model."""
    return MarianTokenizer.from_pretrained(hf_model_name)
