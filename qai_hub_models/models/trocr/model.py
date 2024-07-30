# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import copy
from typing import Callable, List, Tuple

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
TROCR_EXPORT_SEQ_LEN = 1  # -1 TODO(#5428): Dynamic sequence length support. This limits the input size to a seq len of 1.
MODEL_ASSET_VERSION = 1
DEFAULT_NUM_DECODER_LAYERS = 6

"""
Traceable modules used by TrOCRApp
"""
KVCache = Tuple[torch.Tensor, ...]  # Export friendly


class TrOCR(CollectionModel):
    def __init__(
        self,
        encoder: Callable[[torch.Tensor], KVCache],
        decoder: Callable[..., Tuple[torch.Tensor, ...]],
        io_processor: TrOCRProcessor,
        pad_token_id: int,
        eos_token_id: int,
        start_token_id: int,
        max_seq_len: int,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.io_processor = io_processor
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.start_token_id = start_token_id
        self.max_seq_len = max_seq_len

    @classmethod
    def from_pretrained(cls, hf_trocr_model: str = HUGGINGFACE_TROCR_MODEL) -> TrOCR:
        # Load Huggingface source
        source_model = VisionEncoderDecoderModel.from_pretrained(
            hf_trocr_model, return_dict=False
        )
        io_processor = TrOCRProcessor.from_pretrained(hf_trocr_model)
        return TrOCR.from_source_model(source_model, io_processor)  # type: ignore

    @staticmethod
    def from_source_model(
        source_model: VisionEncoderDecoderModel, io_processor: TrOCRProcessor
    ) -> TrOCR:
        encoder = TrOCREncoder(source_model.encoder, source_model.decoder)  # type: ignore
        decoder = TrOCRDecoder(source_model.decoder)  # type: ignore
        return TrOCR(
            encoder,
            decoder,
            io_processor,
            source_model.generation_config.pad_token_id,  # type: ignore
            source_model.generation_config.eos_token_id,  # type: ignore
            (source_model.generation_config.decoder_start_token_id or source_model.generation_config.bos_token_id),  # type: ignore
            source_model.generation_config.max_length,  # type: ignore
        )


class TrOCREncoder(BaseModel):
    """Vision encoder that returns the decoder's cross attn cache state."""

    def __init__(self, encoder: PreTrainedModel, decoder: TrOCRForCausalLM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cross_attn_kv_shape: Callable = decoder.model.decoder.layers[0].encoder_attn._shape  # type: ignore

    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> KVCache:
        """
        Run the encoder on `pixel_values`, and produce a cross attention key/value cache that can be used as decoder input.

        Parameters:
            pixel_values: Pixel values pre-processed for encoder consumption.

        Returns:
            cross_attn_kv_cache: Tuple[kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val, kv_cache_cross_attn_1_key, ...]
                KV Cache for cross attention layers.
                len(tuple) == 2 * number of source model decoder layers.
        """
        encoder_hidden_states = self.encoder(pixel_values, return_dict=False)[0]
        kv_cache = []
        batch_size = encoder_hidden_states.shape[0]
        for layer in self.decoder.model.decoder.layers:
            layer_attn: TrOCRAttention = layer.encoder_attn  # type: ignore
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
    ) -> List[str]:
        output_names = []
        for i in range(num_decoder_layers):
            output_names.append(f"kv_cache_key_{i}")
            output_names.append(f"kv_cache_val_{i}")
        return output_names

    def _get_output_names_for_instance(self) -> List[str]:
        return self.__class__.get_output_names(len(self.decoder.model.decoder.layers))

    @classmethod
    def from_pretrained(cls):
        return TrOCR.from_pretrained().encoder


class TrOCRDecoder(BaseModel):
    """
    Wraps Vision decoder in an export-friendly interface.

    Inputs: (input_ids, KV Cache (unrolled in order generated by combine_kv_caches in app.py))
    Outputs: (output_ids, Updated Attention KV Cache)
    """

    def __init__(self, decoder: TrOCRForCausalLM):
        super().__init__()
        self.decoder = copy.deepcopy(decoder)
        # Delete unused layers that exist only to generate initial KV cache.
        self.num_decoder_layers = len(self.decoder.model.decoder.layers)
        for layer in self.decoder.model.decoder.layers:
            layer_attn: TrOCRAttention = layer.encoder_attn  # type: ignore
            layer_attn.k_proj = None  # type: ignore
            layer_attn.v_proj = None  # type: ignore
        self.max_position_embeddings: int = self.decoder.config.max_position_embeddings  # type: ignore
        self.decoder_attention_heads: int = decoder.config.decoder_attention_heads
        self.embeddings_per_head: int = (
            decoder.config.d_model // decoder.config.decoder_attention_heads
        )

    def forward(
        self, input_ids: torch.IntTensor, *kv_cache_args, **kv_cache_kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generate the next token in the predicted output text sequence.

        Parameters:
            input_ids : torch.IntTensor
                Next token ID in each batch sequence (always shape (batch_size, 1))

            kv_cache: Tuple[kv_cache_attn_0_key, kv_cache_attn_0_val,
                            kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val,
                            kv_cache_attn_1_key, ...]
                Combined KV Cache generated by combine_kv_caches in app.py.
                len(tuple) == 4 * number of source model decoder layers.

        Returns:
            outputs : Tuple[
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
        kv_cache: List[Tuple[torch.Tensor, ...]] = []
        curr_tuple: List[torch.Tensor] = []
        for arg in kv_cache_args or kv_cache_kwargs.values():
            curr_tuple.append(arg)
            if len(curr_tuple) == 4:
                kv_cache.append((*curr_tuple,))
                curr_tuple = []
        kv_cache = (*kv_cache,)  # type: ignore

        # Run decoder
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            use_cache=True,
            past_key_values=kv_cache,
        )

        # KV Cache conversion to export-friendly format (tuple of tensors)
        # Don't output cross attn KV cache because it does not change.
        out_kv_cache: List[torch.Tensor] = []
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
                TROCR_EXPORT_SEQ_LEN,
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

        decoder_input_specs: InputSpec = {"input_ids": input_ids_spec}
        for i in range(0, num_decoder_layers):
            decoder_input_specs[f"kv_{i}_attn_key"] = attn_cache_spec
            decoder_input_specs[f"kv_{i}_attn_val"] = attn_cache_spec
            decoder_input_specs[f"kv_{i}_cross_attn_key"] = cross_attn_cache_spec
            decoder_input_specs[f"kv_{i}_cross_attn_val"] = cross_attn_cache_spec

        return decoder_input_specs

    @staticmethod
    def get_output_names(
        num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS,
    ) -> List[str]:
        output_names = ["next_token"]
        for i in range(num_decoder_layers):
            output_names.append(f"kv_cache_key_{i}")
            output_names.append(f"kv_cache_val_{i}")
        return output_names

    def _get_output_names_for_instance(self) -> List[str]:
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
