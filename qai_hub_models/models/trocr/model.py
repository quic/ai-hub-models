# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.models.trocr.modeling_trocr import (
    PreTrainedModel,
    TrOCRAttention,
    TrOCRForCausalLM,
)

from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

HUGGINGFACE_TROCR_MODEL = "microsoft/trocr-small-stage1"
MODEL_ID = __name__.split(".")[-2]
TROCR_BATCH_SIZE = 1
MAX_DECODE_LEN = 20
MODEL_ASSET_VERSION = 1
DEFAULT_NUM_DECODER_LAYERS = 6

"""
Traceable modules used by TrOCRApp
"""
KVCache = tuple[torch.Tensor, ...]  # Export friendly


class TrOCR(CollectionModel):
    def __init__(
        self,
        encoder: Callable[[torch.Tensor], KVCache],
        decoder: Callable[..., tuple[torch.Tensor, ...]],
        io_processor: TrOCRProcessor,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        start_token_id: int = 2,
        max_seq_len: int = 20,
        num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS,
        decoder_attention_heads: int = 8,
        embeddings_per_head: int = 32,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.io_processor = io_processor
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.start_token_id = start_token_id
        self.max_seq_len = max_seq_len
        self.num_decoder_layers = num_decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.embeddings_per_head = embeddings_per_head

    @classmethod
    def from_pretrained(cls, hf_trocr_model: str = HUGGINGFACE_TROCR_MODEL) -> TrOCR:
        # Load Huggingface source
        source_model = VisionEncoderDecoderModel.from_pretrained(
            hf_trocr_model, return_dict=False
        )
        io_processor = TrOCRProcessor.from_pretrained(hf_trocr_model)

        assert isinstance(source_model, VisionEncoderDecoderModel)
        assert isinstance(io_processor, TrOCRProcessor)
        return TrOCR.from_source_model(source_model, io_processor)

    @staticmethod
    def from_source_model(
        source_model: VisionEncoderDecoderModel, io_processor: TrOCRProcessor
    ) -> TrOCR:
        assert source_model.encoder is not None
        encoder = TrOCREncoder(
            source_model.encoder, source_model.decoder
        )  # pyright: ignore[reportArgumentType]
        decoder = TrOCRDecoder(
            source_model.decoder
        )  # pyright: ignore[reportArgumentType]
        assert source_model.generation_config is not None
        return TrOCR(
            encoder,
            decoder,
            io_processor,
            source_model.generation_config.pad_token_id,
            source_model.generation_config.eos_token_id,
            (
                source_model.generation_config.decoder_start_token_id
                or source_model.generation_config.bos_token_id
            ),
            source_model.generation_config.max_length,
        )


class TrOCREncoder(BaseModel):
    """Vision encoder that returns the decoder's cross attn cache state."""

    def __init__(self, encoder: PreTrainedModel, decoder: TrOCRForCausalLM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cross_attn_kv_shape: Callable = decoder.model.decoder.layers[
            0
        ].encoder_attn._shape

    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> KVCache:
        """
        Run the encoder on `pixel_values`, and produce a cross attention key/value cache that can be used as decoder input.

        Parameters:
            pixel_values: Pixel values pre-processed for encoder consumption.

        Returns:
            cross_attn_kv_cache: tuple[kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val, kv_cache_cross_attn_1_key, ...]
                KV Cache for cross attention layers.
                len(tuple) == 2 * number of source model decoder layers.
        """
        encoder_hidden_states = self.encoder(pixel_values, return_dict=False)[0]
        kv_cache = []
        batch_size = encoder_hidden_states.shape[0]
        for layer in self.decoder.model.decoder.layers:
            layer_attn: TrOCRAttention = layer.encoder_attn
            key_states = self.cross_attn_kv_shape(
                layer_attn.k_proj(encoder_hidden_states), -1, batch_size
            )
            value_states = self.cross_attn_kv_shape(
                layer_attn.v_proj(encoder_hidden_states), -1, batch_size
            )
            kv_cache.append(key_states)
            kv_cache.append(value_states)

        return (*kv_cache,)  # convert list to tuple for export

    @staticmethod
    def get_input_spec(batch_size: int = TROCR_BATCH_SIZE) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declared
        # the model input specification upon submitting a profile job.
        return {"pixel_values": ((batch_size, 3, 384, 384), "float32")}

    @staticmethod
    def get_output_names(
        num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS,
    ) -> list[str]:
        output_names = []
        for i in range(num_decoder_layers):
            output_names.append(f"kv_cache_key_{i}")
            output_names.append(f"kv_cache_val_{i}")
        return output_names

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(len(self.decoder.model.decoder.layers))

    @classmethod
    def from_pretrained(cls):
        return TrOCR.from_pretrained().encoder

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["pixel_values"]


class TrOCRDecoder(BaseModel):
    """
    Wraps Vision decoder in an export-friendly interface.

    Inputs: (input_ids, KV Cache (unrolled in order generated by combine_kv_caches in app.py))
    Outputs: (output_ids, Updated Attention KV Cache)
    """

    def __init__(self, decoder: TrOCRForCausalLM, max_decode_len: int = MAX_DECODE_LEN):
        super().__init__()
        self.decoder = copy.deepcopy(decoder)
        # Delete unused layers that exist only to generate initial KV cache.
        self.num_decoder_layers = len(self.decoder.model.decoder.layers)
        for layer in self.decoder.model.decoder.layers:
            layer_attn: TrOCRAttention = layer.encoder_attn
            layer_attn.k_proj = None  # pyright: ignore[reportAttributeAccessIssue]
            layer_attn.v_proj = None  # pyright: ignore[reportAttributeAccessIssue]
        self.max_position_embeddings: int = self.decoder.config.max_position_embeddings
        self.decoder_attention_heads: int = decoder.config.decoder_attention_heads
        self.embeddings_per_head: int = (
            decoder.config.d_model // decoder.config.decoder_attention_heads
        )

        # Since kv cache is a fixed size, mask out elements
        # that correspond to not yet used entries.
        # The kv cache for current token is inserted at the first
        # index, with the previous cache shifted down by one element.
        #
        # Mask values: 1 for non-padded, 0 for padded
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/trocr/modeling_trocr.py#L799
        self.attn_mask = torch.nn.Embedding(max_decode_len, max_decode_len)
        attn_mask = torch.zeros([max_decode_len, max_decode_len], dtype=torch.float32)
        for c_idx in range(0, max_decode_len):
            attn_mask[c_idx, -(c_idx + 1) :] = 1
        self.attn_mask.weight = torch.nn.Parameter(attn_mask)

    def forward(
        self,
        input_ids: torch.IntTensor,
        index: torch.IntTensor,
        *kv_cache_args,
        **kv_cache_kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Generate the next token in the predicted output text sequence.

        Parameters:
            input_ids : torch.IntTensor
                Next token ID in each batch sequence (always shape (batch_size, 1))

            index: torch.tensor, shape = (batch_size, 1)
                index to get the attn_mask to mask out padded kv cache.

            kv_cache: tuple[kv_cache_attn_0_key, kv_cache_attn_0_val,
                            kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val,
                            kv_cache_attn_1_key, ...]
                Combined KV Cache generated by combine_kv_caches in app.py.
                len(tuple) == 4 * number of source model decoder layers.

        Returns:
            outputs : tuple[
                predicted_ids
                    Next predicted token.
                kv_cache_attn_0_key, kv_cache_attn_0_val, kv_cache_attn_1_key, ...
                    Updated KV cache for attention layers. Count == 2 * number of source model decoder layers.
            ]
        """
        # encoder_hidden_states is not used by the network when kv_cache is set.
        #
        # Unfortunately the underlying huggingface code does not allow us to
        # get rid of the input entirely, because the decoder layer implementation uses its existance
        # to determine if it should include cross-attention layers.
        #
        # Therefore, we set the hidden state to shape [1] in this case to minimize footprint.
        # It will go away when traced.
        encoder_hidden_states = torch.from_numpy(np.array([1]))

        # Convert KV Cache from export friendly format to decoder format
        kv_cache: list[tuple[torch.Tensor, ...]] = []
        curr_tuple: list[torch.Tensor] = []
        for arg in kv_cache_args or kv_cache_kwargs.values():
            curr_tuple.append(arg)
            if len(curr_tuple) == 4:
                kv_cache.append((*curr_tuple,))
                curr_tuple = []
        attn_mask = self.attn_mask(index)
        # (tgt_len,) -> (batch, 1, tgt_len, src_len)

        # Run decoder
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            use_cache=True,
            past_key_values=tuple(kv_cache),
        )

        # KV Cache conversion to export-friendly format (tuple of tensors)
        # Don't output cross attn KV cache because it does not change.
        out_kv_cache: list[torch.Tensor] = []
        for layer_cache in outputs[1]:
            out_kv_cache = out_kv_cache + list(layer_cache)[:2]

        # Argmax Logits, Sequence-Only (Attn) KV Cache
        return (
            torch.argmax(torch.squeeze(outputs[0], dim=1), dim=-1).int(),
            *out_kv_cache,
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = TROCR_BATCH_SIZE,
        decoder_attention_heads: int = 8,
        embeddings_per_head: int = 32,
        num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS,
        max_decode_len: int = MAX_DECODE_LEN,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        input_ids_spec = ((1, 1), "int32")

        attn_cache_spec = (
            (
                batch_size,
                decoder_attention_heads,
                max_decode_len - 1,
                embeddings_per_head,
            ),
            "float32",
        )

        cross_attn_cache_spec = (
            (
                batch_size,
                decoder_attention_heads,
                578,  # TODO: Can we get this programatically?
                embeddings_per_head,
            ),
            "float32",
        )

        decoder_input_specs: InputSpec = {
            "input_ids": input_ids_spec,
            "index": ((1,), "int32"),
        }
        for i in range(0, num_decoder_layers):
            decoder_input_specs[f"kv_{i}_attn_key"] = attn_cache_spec
            decoder_input_specs[f"kv_{i}_attn_val"] = attn_cache_spec
            decoder_input_specs[f"kv_{i}_cross_attn_key"] = cross_attn_cache_spec
            decoder_input_specs[f"kv_{i}_cross_attn_val"] = cross_attn_cache_spec

        return decoder_input_specs

    @staticmethod
    def get_output_names(
        num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS,
    ) -> list[str]:
        output_names = ["next_token"]
        for i in range(num_decoder_layers):
            output_names.append(f"kv_cache_key_{i}")
            output_names.append(f"kv_cache_val_{i}")
        return output_names

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.num_decoder_layers)

    def _get_input_spec_for_instance(
        self, batch_size: int = TROCR_BATCH_SIZE
    ) -> InputSpec:
        return self.__class__.get_input_spec(
            batch_size,
            self.decoder_attention_heads,
            self.embeddings_per_head,
            self.num_decoder_layers,
        )

    @classmethod
    def from_pretrained(cls):
        return TrOCR.from_pretrained().decoder
