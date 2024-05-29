# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import gc
from typing import Any, Callable, List, Tuple

import qai_hub as hub
import torch

from qai_hub_models.models.llama_v2_7b_chat_quantized.model import (
    NUM_SPLITS,
    Llama2_PromptProcessor_1_Quantized,
    Llama2_PromptProcessor_2_Quantized,
    Llama2_PromptProcessor_3_Quantized,
    Llama2_PromptProcessor_4_Quantized,
    Llama2_TokenGenerator_1_Quantized,
    Llama2_TokenGenerator_2_Quantized,
    Llama2_TokenGenerator_3_Quantized,
    Llama2_TokenGenerator_4_Quantized,
    get_input_prompt_with_tags,
    get_past_keyval_with_shift,
    prepare_combined_attention_mask,
)
from qai_hub_models.models.llama_v2_7b_chat_quantized.modeling_llama import (
    RopeEmbedding,
)
from qai_hub_models.utils.base_model import CollectionModel
from qai_hub_models.utils.inference import ExecutableModelProtocol, HubModel
from qai_hub_models.utils.model_adapters import suppress_warnings


def _get_tokens_from_logits(output: torch.Tensor):
    probs = torch.nn.functional.softmax(output[0][0], dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def _get_model_class(split_part: int, is_token_generator: bool = False):
    if split_part < 1 or split_part > 4:
        raise RuntimeError(
            "Incorrect index provided to request Model split class."
            f" Must be within (1-4), provided ({split_part})."
        )

    if is_token_generator:
        return [
            Llama2_TokenGenerator_1_Quantized,
            Llama2_TokenGenerator_2_Quantized,
            Llama2_TokenGenerator_3_Quantized,
            Llama2_TokenGenerator_4_Quantized,
        ][split_part - 1]
    return [
        Llama2_PromptProcessor_1_Quantized,
        Llama2_PromptProcessor_2_Quantized,
        Llama2_PromptProcessor_3_Quantized,
        Llama2_PromptProcessor_4_Quantized,
    ][split_part - 1]


class Llama2ModelPipelineBase(ExecutableModelProtocol):
    """
    Llama Pipeline to execute model splits one after another
    """

    def __init__(self, num_splits: int, is_token_generator: bool = False):
        self.num_splits = num_splits
        self.is_token_generator = is_token_generator
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
            past_key_values.extend(list(out[1:]))

        # Return logits + past_key_values
        return (out[0],) + tuple(past_key_values)

    def forward_tg(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values,
    ):
        past_key_values_new = []
        n = 512
        for i in range(1, self.num_splits + 1):
            with suppress_warnings():
                model = self.load_model_part(i)
            print(f"Running {self.model_type} {i}/{self.num_splits}")
            split_offset = n * (i - 1)
            past_values = past_key_values[split_offset : split_offset + n]
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
                past_j = past_key_values[split_offset + j]

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

        # Return logits + past_key_values
        return (out[0],) + tuple(past_key_values_new)


class HubLlama2ModelPipeline(Llama2ModelPipelineBase):
    """
    Pipeline wrapper for HubModels
    """

    def __init__(
        self,
        hub_model_ids: List[str],
        hub_device: hub.Device,
        inference_options: str,
        is_token_generator: bool = False,
    ):
        super().__init__(len(hub_model_ids), is_token_generator=is_token_generator)
        self.models = []
        for i, model_id in enumerate(hub_model_ids):
            hub_model = HubModel(
                hub.get_model(model_id),
                input_names=_get_model_class(
                    i + 1, is_token_generator=is_token_generator
                )
                .get_input_spec()
                .keys(),
                device=hub_device,
                inference_options=inference_options,
                output_names=_get_model_class(
                    i + 1, is_token_generator=is_token_generator
                ).get_output_names(),
            )
            self.models.append(hub_model)

    def load_model_part(self, model_part: int):
        model_index = model_part - 1
        if model_index < 0 or model_index > len(self.models):
            raise RuntimeError(
                f"HubLlama2ModelPipeline does not have requested model_part {model_part}."
            )

        return self.models[model_index]


class Llama2ModelPipeline(Llama2ModelPipelineBase):
    """
    Pipeline wrapper for PyTorch base model
    """

    def __init__(
        self, prompt_processor: CollectionModel, is_token_generator: bool = False
    ):
        self.prompt_processor = prompt_processor
        self.model_type = "TokenGenerator" if is_token_generator else "PromptProcessor"
        super().__init__(NUM_SPLITS, is_token_generator=is_token_generator)

    def load_model_part(self, model_part: int):
        if model_part < 1 or model_part > NUM_SPLITS:
            raise RuntimeError(
                f"ModelLlama2ModelPipeline does not have requested model_part {model_part}."
            )
        return self.prompt_processor.load_model_part(
            f"Llama2_{self.model_type}_{model_part}_Quantized"
        )


class ChatApp:
    def __init__(
        self, prompt_processor: Callable, token_generator: Callable, tokenizer: Any
    ):
        self.prompt_processor = prompt_processor
        self.token_generator = token_generator
        self.tokenizer = tokenizer

    def generate_output_prompt(
        self, input_prompt: str, max_seq_len: int, max_output_tokens: int
    ):
        input_prompt_processed = get_input_prompt_with_tags(
            user_input_prompt=input_prompt
        )
        input_tokens = self.tokenizer(input_prompt_processed, return_tensors="pt")
        token_size = input_tokens["input_ids"].shape[-1]
        padding_size = max_seq_len - token_size

        input_ids = torch.cat(
            (
                torch.Tensor([self.tokenizer.unk_token_id] * padding_size).reshape(
                    1, padding_size
                ),
                input_tokens["input_ids"],
            ),
            dim=-1,
        ).type(torch.int32)
        attention_mask = torch.cat(
            (
                torch.Tensor([0] * padding_size).reshape(1, padding_size),
                input_tokens["attention_mask"],
            ),
            dim=-1,
        ).type(torch.int32)
        cm_attention_masks = prepare_combined_attention_mask(
            attention_mask=attention_mask
        )
        position_ids = (
            torch.cat(
                (
                    torch.zeros(
                        padding_size,
                    ),
                    torch.arange(token_size),
                )
            )
            .reshape(1, max_seq_len)
            .type(torch.int32)
        )

        position_ids = (
            torch.Tensor(position_ids).type(torch.long).reshape(1, max_seq_len)
        )
        position_ids_cos, position_ids_sin = RopeEmbedding(
            max_length=max_seq_len
        ).get_embedding(position_ids)
        output = self.prompt_processor(
            input_ids, cm_attention_masks, position_ids_cos, position_ids_sin
        )
        output_token = _get_tokens_from_logits(output)
        past_key_values = get_past_keyval_with_shift(output[1:]).values()
        output_prompt = self.tokenizer.decode(output_token)
        print()
        print(f"Text generated by Prompt Processor: {output_prompt}")
        print()

        # Collect output prompt to summarize later
        hub_tokens = output_token
        num_of_tokens_processed = token_size + 1

        # TODO: Revisiting demo and app to refactor like a chat-bot
        # This is just a place-holder to show how both models work together
        for _ in range(max_output_tokens - 1):
            # TODO: check if previous generated token is EOS
            if num_of_tokens_processed >= max_seq_len:
                break

            input_ids = output_token.reshape(-1, 1).type(torch.int32)
            # Shift attention_mask and position_ids
            attention_mask = torch.cat(
                (attention_mask[:, 1:], torch.Tensor([[1]])), dim=-1
            )
            cm_attention_masks = prepare_combined_attention_mask(
                attention_mask=attention_mask,
                input_shape=(1, 1),
                past_key_values_length=max_seq_len - 1,
            )
            position_ids = (position_ids[:, -1] + 1).reshape(-1, 1)

            position_ids = torch.Tensor(position_ids).type(torch.long).reshape(1, 1)
            position_ids_cos, position_ids_sin = RopeEmbedding(
                max_length=max_seq_len
            ).get_embedding(position_ids)
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
        return output_prompt
