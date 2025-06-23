# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import math
from collections.abc import Generator
from typing import Union

import torch
import transformers
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.models._shared.llm.model import Embedding, LLM_AIMETOnnx


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """
    Clip past key value to feed next iteration
    """
    ret = []

    if len(past_key_vals) == 0:
        for i in range(0, len(new_key_vals), 2):
            key_shape = new_key_vals[i].shape
            key_shape = (key_shape[0], key_shape[1], key_shape[2], 0)
            past_key_vals.append(torch.zeros(key_shape, device=device))

            value_shape = new_key_vals[i + 1].shape
            value_shape = (value_shape[0], value_shape[1], 0, value_shape[3])
            past_key_vals.append(torch.zeros(value_shape, device=device))

    if len(new_key_vals) == 0:
        for i in range(0, len(past_key_vals), 2):
            key_shape = past_key_vals[i].shape
            key_shape = (key_shape[0], key_shape[1], key_shape[2], 0)
            new_key_vals.append(torch.zeros(key_shape, device=device))

            value_shape = past_key_vals[i + 1].shape
            value_shape = (value_shape[0], value_shape[1], 0, value_shape[3])
            new_key_vals.append(torch.zeros(value_shape, device=device))

    # Key and Values are concatenated on batch dimension
    for i in range(0, len(past_key_vals), 2):
        key_cache = torch.cat(
            [past_key_vals[i].to(device), new_key_vals[i].to(device)],
            dim=3,
        )
        key_cache = key_cache[:, :, :, -length:]
        val_cache = torch.cat(
            [
                past_key_vals[i + 1].to(device),
                new_key_vals[i + 1].to(device),
            ],
            dim=2,
        )
        val_cache = val_cache[:, :, -length:, :]

        ret.append(key_cache)
        ret.append(val_cache)
    return ret


class LLM_Loader:
    def __init__(
        self,
        model_cls: type[LLM_AIMETOnnx],
        sequence_length: int,
        model_params,
        host_device: torch.device,
    ):
        self.model_cls = model_cls
        self.sequence_length = sequence_length
        self.model_params = model_params
        self.loaded_model = None
        self.host_device = host_device

    def load(self) -> LLM_AIMETOnnx:
        if self.loaded_model is None:
            self.loaded_model = self.model_cls.from_pretrained(
                sequence_length=self.sequence_length, **self.model_params
            ).to(self.host_device)

        assert self.loaded_model is not None
        return self.loaded_model

    def release(self):
        self.loaded_model = None


class LLM_Generator(GenerationMixin, torch.nn.Module):
    def __init__(
        self,
        models: list[Union[LLM_AIMETOnnx, LLM_Loader]],
        tokenizer: transformers.PreTrainedTokenizer,
        embedding: Embedding,
    ):
        super().__init__()

        self.models = models
        self.models.sort(key=lambda model: model.sequence_length)

        self.selected_model = (
            self.models[-1].load()
            if isinstance(self.models[-1], LLM_Loader)
            else self.models[-1]
        )

        self.tokenizer = tokenizer
        self.embedding = embedding

        self.generation_config = None

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.selected_model.llm_config

    @property
    def main_input_name(self) -> str:
        return "input_ids"

    @property
    def _supports_cache_class(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        if hasattr(self.selected_model, "host_device"):
            # Only works for models derived from LLM_AIMETOnnx
            return self.selected_model.host_device
        else:
            return self.selected_model.model.device

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | DynamicCache | None]:

        """
        Overridden prepare_inputs_for_generation function to enable Huggingface generate() on models with static
        graph constraints
        """

        # We need a way to ensure that all the previous tokens that have already been consumed are stripped out of the
        # input ids

        # If past_key_values is None, this indicates that this `prepare_inputs_for_generation()` is being called for
        # the first time, and nothing should be stripped out of `input_ids`. In other cases though, the number of tokens
        # already inside `past_key_values` indicates how many tokens should be stripped out of `input_ids`

        # Notes: `input_ids`, `attention_mask`, `past_key_values` should NOT have static shape requirements imposed on
        # them by the time they reach this function. That is, in order for this to work, the static shape padding and
        # truncation must happen directly in the model `forward` function

        num_processed_tokens = (
            0
            if past_key_values is None
            or len(past_key_values.value_cache) == 0
            or past_key_values.value_cache[0] == []
            else past_key_values.value_cache[0].shape[-2]
        )
        return {
            "input_ids": input_ids[:, num_processed_tokens:],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    def select_model(self, num_input_ids):
        # Select the model with the smallest sequence length that can fit all of num_input_ids
        # If there is no model that can consume num_input_ids in one inference, select the model with the largest
        # context length
        def find_model(num_input_ids):
            for model in self.models:
                if num_input_ids <= model.sequence_length:
                    return model
            return self.models[-1]

        # update selected model based on num_tokens to consume
        new_selected_model = find_model(num_input_ids)

        if self.selected_model.sequence_length == new_selected_model.sequence_length:
            return self.selected_model

        print(
            f"Switching from sequence_length={self.selected_model.sequence_length} to sequence_length={new_selected_model.sequence_length}"
        )
        # release all LLM_Loader objects to preserve memory
        if isinstance(self.selected_model, LLM_Loader):
            self.selected_model.release()
        if isinstance(self.selected_model, LLM_AIMETOnnx):
            del self.selected_model.quant_sim

        self.selected_model = (
            new_selected_model.load()
            if isinstance(new_selected_model, LLM_Loader)
            else new_selected_model
        )
        return self.selected_model

    @staticmethod
    def slice_inputs_for_inference(
        input_ids: torch.Tensor, attention_mask: torch.Tensor, sequence_length: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        input_length = input_ids.shape[-1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            yield input_ids[:, max(0, idx - sequence_length) : idx], attention_mask[
                :, max(0, idx - sequence_length) : idx
            ]

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
    ) -> tuple[torch.Tensor, ...]:
        device = input_ids.device
        batch_size, input_length = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        input_ids_extension = torch.full(
            (batch_size, sequence_length - input_length),
            fill_value=getattr(self.tokenizer, "eos_token_id", 0),
            dtype=input_ids.dtype,
            device=device,
        )
        padded_input_ids = torch.cat((input_ids_extension, input_ids), dim=-1)
        padded_attention_mask = torch.cat(
            (torch.zeros_like(input_ids_extension), attention_mask), dim=-1
        )

        input_specs = self.selected_model.get_input_spec(
            sequence_length=sequence_length,
            context_length=context_length,
        )
        # Initialization of KV cache padding
        dummy_past_key_values = [
            torch.zeros(shape, device=device)
            for k, (shape, _) in input_specs.items()
            if k.startswith("past_")
        ]

        current_key_value_length = (
            past_key_values[0].shape[-1] if past_key_values else 0
        )
        key_value_padding_length = (
            context_length - sequence_length
        ) - current_key_value_length

        padded_past_key_values = get_past_keyval_with_shift(
            past_key_vals=dummy_past_key_values,
            new_key_vals=past_key_values,
            length=context_length - sequence_length,
            device=device,
        )

        kv_cache_attention_mask = torch.cat(
            (
                torch.zeros((batch_size, key_value_padding_length)),
                torch.ones((batch_size, current_key_value_length)),
            ),
            dim=-1,
        ).to(device=device)
        padded_attention_mask = torch.cat(
            (kv_cache_attention_mask, padded_attention_mask), dim=-1
        )

        position_ids = torch.cumsum(padded_attention_mask, dim=1, dtype=torch.int32) - 1
        position_ids = position_ids.clip(0, context_length - 1)
        position_ids = position_ids[..., -sequence_length:]
        position_ids_cos, position_ids_sin = self.embedding.get_embedding(position_ids)

        attention_mask_converter = AttentionMaskConverter(True)
        cm_attention_mask = attention_mask_converter.to_4d(
            padded_attention_mask,
            query_length=sequence_length,
            key_value_length=context_length,
            dtype=torch.float32,
        )
        cm_attention_mask = cm_attention_mask.clip(-50, 0)

        return (
            padded_input_ids.to(torch.int32),
            cm_attention_mask,
            position_ids_cos,
            position_ids_sin,
            *padded_past_key_values,
        )

    def combine_local_and_global_outputs(
        self,
        model: LLM_AIMETOnnx,
        num_valid_input_tokens: int,
        local_outputs: tuple[torch.Tensor, ...],
        global_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]],
    ):
        # strip logits corresponding to padding tokens
        local_logits = local_outputs[0]
        local_logits = torch.narrow(
            local_logits,
            1,
            local_logits.shape[1] - num_valid_input_tokens,
            num_valid_input_tokens,
        )

        # concatenate logits from local inference to global output
        global_outputs["logits"] = (
            torch.cat((global_outputs["logits"], local_logits), dim=1)
            if "logits" in global_outputs
            else local_logits
        )

        # strip KV cache corresponding to padding tokens
        local_past_key_values = get_past_keyval_with_shift(
            past_key_vals=[],
            new_key_vals=list(local_outputs[1:]),
            length=num_valid_input_tokens,
            device=torch.device("cpu"),
        )

        # shift global KV cache, concatenate local KV cache
        current_key_value_length = (
            global_outputs["past_key_values"][0].shape[-1]
            if global_outputs["past_key_values"]
            else 0
        )
        global_outputs["past_key_values"] = get_past_keyval_with_shift(
            past_key_vals=global_outputs["past_key_values"],
            new_key_vals=local_past_key_values,
            length=min(
                current_key_value_length + num_valid_input_tokens,
                model.context_length - model.sequence_length,
            ),
            device=torch.device("cpu"),
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: DynamicCache = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Select which model to use
        model = self.select_model(input_ids.shape[-1])

        # Create attention mask if one does not exist
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        global_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]] = {
            "past_key_values": []
            if past_key_values is None or past_key_values.get_seq_length() == 0
            else list(itertools.chain.from_iterable(past_key_values.to_legacy_cache()))
        }

        for input_ids_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_ids, attention_mask, model.sequence_length
        ):
            prepared_inputs = self.prepare_inputs(
                input_ids_slice,
                attention_mask_slice,
                global_outputs["past_key_values"],
                model.sequence_length,
                model.context_length,
            )

            local_outputs = model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                model, input_ids_slice.shape[-1], local_outputs, global_outputs
            )

        # make sure logits are on the correct device (necessary for generation)
        # the underlying mock_torch_onnx_inference function does not necessarily move outputs back to CUDA
        assert isinstance(global_outputs["logits"], torch.Tensor)
        logits = global_outputs["logits"].to(device=input_ids.device)

        # Convert KV Cache outputs into HF DynamicCache
        past_key_values = DynamicCache()
        past_key_values.key_cache = global_outputs["past_key_values"][::2]
        past_key_values.value_cache = global_outputs["past_key_values"][1::2]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def prefill(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: DynamicCache = None,
        **kwargs,
    ) -> Generator[tuple[torch.Tensor, ...]]:
        if len(self.models) > 1:
            raise RuntimeError("Prefill should only be invoked using a single model")

        # Select which model to use
        model = self.select_model(input_ids.shape[-1])

        # Create attention mask if one does not exist
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # slice input ids and attention mask to drop last few tokens
        total_num_inferences = math.ceil(input_ids.shape[-1] / model.sequence_length)
        num_tokens_to_preconsume = (total_num_inferences - 1) * model.sequence_length

        input_ids_to_preconsume = input_ids[:, :num_tokens_to_preconsume]
        attention_mask_to_preconsume = attention_mask[:, :num_tokens_to_preconsume]

        preconsumed_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]] = {
            "past_key_values": []
            if past_key_values is None or past_key_values.get_seq_length() == 0
            else list(itertools.chain.from_iterable(past_key_values.to_legacy_cache()))
        }

        for input_ids_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_ids_to_preconsume, attention_mask_to_preconsume, model.sequence_length
        ):
            prepared_inputs = self.prepare_inputs(
                input_ids_slice,
                attention_mask_slice,
                preconsumed_outputs["past_key_values"],
                model.sequence_length,
                model.context_length,
            )

            yield tuple(tensor.cpu() for tensor in prepared_inputs)

            local_outputs = model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                model, input_ids_slice.shape[-1], local_outputs, preconsumed_outputs
            )

        remaining_input_ids = input_ids[:, num_tokens_to_preconsume:]
        remaining_attention_mask = attention_mask[:, num_tokens_to_preconsume:]
        prefilled_inputs = self.prepare_inputs(
            remaining_input_ids,
            remaining_attention_mask,
            preconsumed_outputs["past_key_values"],
            model.sequence_length,
            model.context_length,
        )

        yield tuple(tensor.cpu() for tensor in prefilled_inputs)
