# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import gc
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Set, Tuple

import qai_hub as hub
import torch

from qai_hub_models.models._shared.llama.model import (
    RopeEmbedding,
    get_past_keyval_with_shift,
)
from qai_hub_models.utils.base_model import CollectionModel
from qai_hub_models.utils.inference import ExecutableModelProtocol, OnDeviceModel
from qai_hub_models.utils.model_adapters import suppress_warnings


def _get_tokens_from_logits(output: torch.Tensor):
    probs = torch.nn.functional.softmax(output[0][0], dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


class LlamaModelPipelineBase(ExecutableModelProtocol):
    """
    Llama Pipeline to execute model splits one after another
    """

    def __init__(
        self,
        num_splits: int,
        num_past_key_val_heads: int,
        model_split_map: Dict[int, Tuple[int, int]],
        is_token_generator: bool = False,
    ):
        self.num_splits = num_splits
        self.is_token_generator = is_token_generator
        self.num_past_key_val_heads = num_past_key_val_heads
        self.model_split_map = model_split_map
        self.model_type = "TokenGenerator" if is_token_generator else "PromptProcessor"

    def __call__(
        self,
        *args: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        if self.is_token_generator:
            return self.forward_tg(*args)
        return self.forward(*args)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
    ):
        past_key_values = []
        for i in range(1, self.num_splits + 1):
            with suppress_warnings():
                model = self.load_model_part(i)
            print(f"Running {self.model_type} {i}/{self.num_splits}")
            out = model(input_ids, attention_mask, position_ids_cos, position_ids_sin)
            out = [x.detach() for x in out]
            # free model to reduce memory-pressure
            # NOTE: quantized models should not need this.
            del model
            gc.collect()
            input_ids = out[0]
            for each in out[1:]:
                past_key_values.extend(list(torch.split(each, 1, dim=1)))

        # Return logits + past_key_values
        return tuple((out[0], *past_key_values))

    def forward_tg(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values,
    ):
        past_key_values_new = []
        start_past_key_offset = 0
        for i in range(1, self.num_splits + 1):
            with suppress_warnings():
                model = self.load_model_part(i)
            print(f"Running {self.model_type} {i}/{self.num_splits}")
            layer_start, layer_end = self.model_split_map[i]
            num_of_key_vals = (
                self.num_past_key_val_heads * 2 * (layer_end - layer_start)
            )

            end_past_key_offset = start_past_key_offset + num_of_key_vals
            past_values = past_key_values[start_past_key_offset:end_past_key_offset]
            out = model(
                input_ids,
                attention_mask,
                position_ids_cos,
                position_ids_sin,
                *past_values,
            )
            out = [x.detach() for x in out]
            # free model to reduce memory-pressure
            # NOTE: quantized models should not need this.
            del model
            gc.collect()
            input_ids = out[0]

            for j, new_cache_j in enumerate(out[1:]):
                # Construct new past entries by concatenating old and new
                past_j = past_key_values[start_past_key_offset + j]

                # Concatenation is not always along the same dimension
                if new_cache_j.shape[3] == 1:
                    dim = 3
                else:
                    dim = 2

                # Slice to remove oldest value
                slices = [slice(None)] * dim + [slice(1, None)]

                past_key_values_new.append(
                    torch.cat(
                        [
                            past_j[slices],
                            new_cache_j,
                        ],
                        dim=dim,
                    )
                )
            start_past_key_offset = end_past_key_offset

        # Return logits + past_key_values
        return tuple((out[0], *past_key_values_new))

    @abstractmethod
    def load_model_part(self, model_part: int):
        pass


class OnDeviceLlamaModelPipeline(LlamaModelPipelineBase):
    """
    Pipeline wrapper for OnDeviceModels
    """

    def __init__(
        self,
        hub_model_ids: List[str],
        hub_device: hub.Device,
        inference_options: str,
        get_model_class: Callable,
        num_past_key_val_heads: int,
        model_split_map: Dict[int, Tuple[int, int]],
        is_token_generator: bool = False,
    ):
        super().__init__(
            len(hub_model_ids),
            num_past_key_val_heads,
            model_split_map,
            is_token_generator=is_token_generator,
        )
        self.models = []
        for i, model_id in enumerate(hub_model_ids):
            hub_model = OnDeviceModel(
                hub.get_model(model_id),
                input_names=get_model_class(
                    i + 1, is_token_generator=is_token_generator
                )
                .get_input_spec()
                .keys(),
                device=hub_device,
                inference_options=inference_options,
                output_names=get_model_class(
                    i + 1, is_token_generator=is_token_generator
                ).get_output_names(),
            )
            self.models.append(hub_model)

    def load_model_part(self, model_part: int):
        model_index = model_part - 1
        if model_index < 0 or model_index > len(self.models):
            raise RuntimeError(
                f"HubLlamaModelPipeline does not have requested model_part {model_part}."
            )
        return self.models[model_index]


class LlamaModelPipeline(LlamaModelPipelineBase):
    """
    Pipeline wrapper for PyTorch base model
    """

    def __init__(
        self,
        models: CollectionModel,
        num_splits: int,
        num_past_key_val_heads: int,
        model_split_map: Dict[int, Tuple[int, int]],
        is_token_generator: bool = False,
    ):
        self.models = models
        self.num_splits = num_splits
        self.model_type = "TokenGenerator" if is_token_generator else "PromptProcessor"
        super().__init__(
            num_splits,
            num_past_key_val_heads=num_past_key_val_heads,
            model_split_map=model_split_map,
            is_token_generator=is_token_generator,
        )

    def load_model_part(self, model_part: int):
        if model_part < 1 or model_part > self.num_splits:
            raise RuntimeError(
                f"ModelLlamaModelPipeline does not have requested model_part {model_part}."
            )
        return self.models.load_model_part(f"{self.model_type}_{model_part}_Quantized")


class ChatApp:
    """
    This class is demonstration of how once can use Llama model to build a basic ChatApp.
    This App use two models
        * Prompt Processor
            - Processes user input prompt to generate first token and KV-cache
        * Token Generator
            - Generators output token one at a time
            - Uses KV-cache to speed up token generation
    """

    def __init__(
        self,
        prompt_processor: Callable,
        token_generator: Callable,
        get_input_prompt_with_tags: Callable,
        prepare_combined_attention_mask: Callable,
        tokenizer: Any,
        end_tokens: Set[str],
        num_past_key_val_heads: int,
    ):
        """
        Base ChatApp that generates one response for given input token.

            prompt_processor: Prompt Processor collection model
            token_generator: Token Generator collection model
            get_input_prompt_with_tags: Function to wrap input prompt with appropriate tags
            prepare_combined_attention_mask: Function to combine and build attention mask,
            tokenizer: Tokenizer to use,
            end_tokens: Set of end tokens to convey end of token generation,
            num_past_key_val_heads: Number of heads in past-key value,
        """
        self.prompt_processor = prompt_processor
        self.token_generator = token_generator
        self.get_input_prompt_with_tags = get_input_prompt_with_tags
        self.prepare_combined_attention_mask = prepare_combined_attention_mask
        self.tokenizer = tokenizer
        self.end_tokens = end_tokens
        self.num_past_key_val_heads = num_past_key_val_heads

    def generate_output_prompt(
        self, input_prompt: str, max_seq_len: int, max_output_tokens: int
    ):
        input_prompt_processed = self.get_input_prompt_with_tags(
            user_input_prompt=input_prompt
        )

        input_tokens = self.tokenizer(
            input_prompt_processed,
            return_tensors="pt",
            padding="max_length",
            max_length=max_seq_len,
        )
        input_ids = input_tokens["input_ids"].type(torch.long)
        num_tokens = torch.sum(input_tokens["attention_mask"]).item()
        padding_size = max_seq_len - num_tokens
        position_ids = [0] * (padding_size) + list(range(0, num_tokens))
        position_ids = (
            torch.Tensor(position_ids).type(torch.long).reshape(1, max_seq_len)
        )
        attention_mask = input_tokens["attention_mask"].type(torch.float32)
        cm_attention_masks = self.prepare_combined_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_tokens["attention_mask"].shape,
        )
        position_ids = (
            torch.Tensor(position_ids).type(torch.long).reshape(1, max_seq_len)
        )
        position_ids_cos, position_ids_sin = RopeEmbedding(
            max_length=max_seq_len
        ).get_embedding(position_ids)

        # Process input prompt
        output = self.prompt_processor(
            input_ids, cm_attention_masks, position_ids_cos, position_ids_sin
        )
        output_token = _get_tokens_from_logits(output)
        past_key_values = get_past_keyval_with_shift(
            output[1:], num_of_past_key_heads=self.num_past_key_val_heads
        ).values()
        output_prompt = self.tokenizer.decode(output_token)
        print()
        print(f"Text generated by Prompt Processor: {output_prompt}")
        print()

        # Collect output prompt to summarize later
        hub_tokens = output_token
        num_of_tokens_processed = num_tokens + 1

        for _ in range(max_output_tokens - 1):
            # TODO: check if previous generated token is EOS
            if num_of_tokens_processed >= max_seq_len:
                break

            input_ids = output_token.reshape(-1, 1).type(torch.int32)
            # Shift attention_mask and position_ids
            attention_mask = torch.cat(
                (attention_mask[:, 1:], torch.Tensor([[1]])), dim=-1
            )
            cm_attention_masks = self.prepare_combined_attention_mask(
                attention_mask=attention_mask,
                input_shape=(1, 1),
                past_key_values_length=max_seq_len - 1,
            )
            position_ids = (position_ids[:, -1] + 1).reshape(-1, 1)

            position_ids = torch.Tensor(position_ids).type(torch.long).reshape(1, 1)
            position_ids_cos, position_ids_sin = RopeEmbedding(
                max_length=max_seq_len
            ).get_embedding(position_ids)

            # Generate output token
            output = self.token_generator(
                input_ids,
                cm_attention_masks,
                position_ids_cos,
                position_ids_sin,
                *past_key_values,
            )
            del cm_attention_masks
            del input_ids
            output_token = _get_tokens_from_logits(output)
            output_prompt = self.tokenizer.decode(output_token)

            # Assistant generating end of token
            if output_prompt in self.end_tokens:
                break

            past_key_values = output[1:]
            hub_tokens = torch.cat((hub_tokens, output_token), dim=-1)
            print()
            print(f"Text generated so far: {self.tokenizer.decode(hub_tokens)}")
            print()
            num_of_tokens_processed += 1
            gc.collect()

        print("-------- Response Summary --------")
        print(f"Prompt: {input_prompt}")
        print(f"Response: {self.tokenizer.decode(hub_tokens)}")
