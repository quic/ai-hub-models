# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import gc
import math
from collections.abc import Callable
from typing import Any

import torch
from transformers.models.llama import LlamaConfig, modeling_llama

from qai_hub_models.utils.base_model import BaseModel


class RopeEmbedding:
    def __init__(
        self,
        head_dim: int = 128,
        max_length: int = 2048,
        config: LlamaConfig = LlamaConfig(),
    ) -> None:
        self.cos, self.sin = self.precompute(head_dim, max_length, config)

    def precompute(
        self, head_dim: int, max_length: int, config: LlamaConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        kwargs = {
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "config": config,
        }

        if not hasattr(config, "rope_scaling"):
            setattr(config, "rope_scaling", None)

        rope = modeling_llama.LlamaRotaryEmbedding(dim=head_dim, **kwargs)
        dummy_x = torch.Tensor([1.0])
        position_ids = torch.arange(max_length).view(1, -1)
        if hasattr(rope, "_original_forward"):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        embeddings = [emb.unsqueeze(0) for emb in embeddings]
        return embeddings  # pyright: ignore [reportReturnType]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


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

    # Key and Values are concatanated on batch dimension
    for i in range(0, len(past_key_vals), 2):
        n = new_key_vals[i].shape[3]
        m = past_key_vals[i].shape[3]
        remove = n + m - length
        key_cache = torch.cat(
            [past_key_vals[i][:, :, :, remove:].to(device), new_key_vals[i].to(device)],
            dim=3,
        )
        val_cache = torch.cat(
            [
                past_key_vals[i + 1][:, :, remove:].to(device),
                new_key_vals[i + 1].to(device),
            ],
            dim=2,
        )

        ret.append(key_cache)
        ret.append(val_cache)
    return ret


def _sample_tokens_from_logits(
    logits: torch.Tensor, top_k: int = 40, top_p: float = 0.95, temp: float = 0.8
) -> torch.Tensor:
    assert logits.ndim == 2

    values, indices = torch.topk(logits, top_k, sorted=True)

    probs = torch.nn.functional.softmax(values, dim=-1)

    is_cut_off = torch.cumsum(probs, dim=-1) > top_p
    if is_cut_off.any():
        cut_off_index = torch.nonzero(is_cut_off)[0, 1].item()
        values = values[:, : cut_off_index + 1]
        indices = indices[:, : cut_off_index + 1]

    probs = torch.nn.functional.softmax(values / temp, dim=-1)

    inner_index = torch.multinomial(probs, num_samples=1).squeeze(1)
    return indices[0][inner_index[0].item()].unsqueeze(0)


class ChatApp:
    """
    This class is a demonstration of how to use Llama model to build a basic ChatApp.
    This App uses two models:
        * Prompt Processor
            - Instantiation with sequence length 128. Used to process user
              prompt.
        * Token Generator
            - Instantiation with sequence length 1. Used to predict
              auto-regressive response.
    """

    def __init__(
        self,
        model_cls: type[BaseModel],
        get_input_prompt_with_tags: Callable,
        prepare_combined_attention_mask: Callable,
        tokenizer: Any,
        end_tokens: set[str],
        seed: int = 42,
    ):
        """
        Base ChatApp that generates one response for given input token.

            model_cls: Llama Model class that will be used to instantiate model
            get_input_prompt_with_tags: Function to wrap input prompt with appropriate tags
            prepare_combined_attention_mask: Function to combine and build attention mask,
            tokenizer: Tokenizer to use,
            end_tokens: Set of end tokens to convey end of token generation,
        """
        self.model_cls = model_cls
        self.get_input_prompt_with_tags = get_input_prompt_with_tags
        self.prepare_combined_attention_mask = prepare_combined_attention_mask
        self.tokenizer = tokenizer
        self.end_tokens = end_tokens
        self.seed = seed

    def generate_output_prompt(
        self,
        input_prompt: str,
        prompt_sequence_length: int,
        context_length: int,
        max_output_tokens: int,
        checkpoint: str | None = None,
        bundled_kvcache: bool = True,
        model_from_pretrained_extra: dict = {},
    ):
        torch.manual_seed(self.seed)
        input_prompt_processed = self.get_input_prompt_with_tags(
            user_input_prompt=input_prompt
        )

        input_tokens = self.tokenizer(
            input_prompt_processed,
            return_tensors="pt",
            padding="max_length",
            max_length=context_length,
        )
        if context_length % prompt_sequence_length != 0:
            raise ValueError(
                "This script requires the prompt sequence lengths to evenly divide the context length."
            )

        host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        orig_input_ids = input_tokens["input_ids"].type(torch.long).to(host_device)

        num_tokens = int(torch.sum(input_tokens["attention_mask"]).item())
        num_prompt_iterations = math.ceil(num_tokens / prompt_sequence_length)

        print(
            f"Will run prompt processor {num_prompt_iterations} time(s) and then token generator."
        )

        # Collect output prompt to summarize later
        output_token = None
        hub_tokens: torch.Tensor | None = None

        model_params = {
            "context_length": context_length,
            "host_device": host_device,
            **model_from_pretrained_extra,
        }
        if checkpoint is not None:
            model_params["checkpoint"] = checkpoint

        model = self.model_cls.from_pretrained(
            sequence_length=prompt_sequence_length,
            **model_params,
        )
        rope_embedding = RopeEmbedding(
            max_length=context_length, config=model.llm_config
        )
        is_prompt = True

        # Process input prompt
        input_specs = self.model_cls.get_input_spec(
            sequence_length=prompt_sequence_length,
            context_length=context_length,
        )

        # Initialization of KV cache
        past_key_values = [
            torch.zeros(shape, device=host_device)
            for k, (shape, _) in input_specs.items()
            if k.startswith("past_")
        ]

        position_ids: torch.Tensor | None = None
        attention_mask: torch.Tensor | None = None
        for i in range(num_prompt_iterations + max_output_tokens - 1):
            if i < num_prompt_iterations:
                seq_len = prompt_sequence_length
                next_seq_len = seq_len if i + 1 < num_prompt_iterations else 1
            else:
                if is_prompt:
                    # Make sure GPU memory is freed
                    if hasattr(model, "quant_sim"):
                        del model.quant_sim

                    # switch to token processor
                    model = self.model_cls.from_pretrained(
                        sequence_length=1,
                        **model_params,
                    )
                    is_prompt = False

                seq_len = 1
                next_seq_len = 1

            if is_prompt:
                input_ids = orig_input_ids[
                    :,
                    max(
                        0, context_length - (num_prompt_iterations - i) * seq_len
                    ) : max(
                        0, context_length - (num_prompt_iterations - i - 1) * seq_len
                    ),
                ]

                # non-padded tokens in first prompt
                first_prompt = (num_tokens - 1) % seq_len + 1
                padding_size0 = seq_len - first_prompt
                padding_size = padding_size0 if i == 0 else 0
                offset = 0 if i == 0 else first_prompt + (i - 1) * seq_len
                position_ids_lst = [0] * (padding_size) + list(
                    range(offset, offset + seq_len - padding_size)
                )
                position_ids = (
                    torch.Tensor(position_ids_lst)
                    .type(torch.long)
                    .reshape(1, seq_len)
                    .to(host_device)
                )
                position_ids_cos, position_ids_sin = rope_embedding.get_embedding(
                    position_ids,
                )
                attention_mask = torch.zeros((1, context_length), device=host_device)
                attention_mask[:, context_length - (first_prompt + i * seq_len) :] = 1.0
            else:
                assert output_token is not None
                input_ids = output_token.reshape(-1, 1).type(torch.int32)

                # Shift attention_mask and position_ids
                assert attention_mask is not None
                attention_mask = torch.cat(
                    (
                        attention_mask[:, seq_len:],
                        torch.ones((1, seq_len), device=host_device),
                    ),
                    dim=-1,
                )
                assert position_ids is not None
                position_ids = (position_ids[:, -1] + 1).reshape(-1, 1)

                position_ids = (
                    torch.Tensor(position_ids)
                    .type(torch.long)
                    .reshape(1, 1)
                    .to(host_device)
                )
                position_ids_cos, position_ids_sin = rope_embedding.get_embedding(
                    position_ids,
                )

            cm_attention_masks = self.prepare_combined_attention_mask(
                attention_mask=attention_mask,
                input_shape=(1, seq_len),
                past_key_values_length=context_length - seq_len,
            ).to(host_device)

            # Generate output token
            output = model(
                input_ids.to(torch.int32),
                cm_attention_masks,
                position_ids_cos,
                position_ids_sin,
                *past_key_values,
            )

            del cm_attention_masks
            del input_ids
            past_key_values = get_past_keyval_with_shift(
                past_key_values,
                output[1:],
                length=context_length - next_seq_len,
                device=host_device,
            )
            is_prediction = next_seq_len == 1

            if is_prediction:
                # Sample output
                output_token = _sample_tokens_from_logits(output[0][0][[-1]])

                # Assistant generating end of token
                if self.tokenizer.decode(output_token) in self.end_tokens:
                    break

                if is_prompt:
                    hub_tokens = output_token
                else:
                    assert hub_tokens is not None
                    hub_tokens = torch.cat((hub_tokens, output_token), dim=-1)

                print()
                print(f"Text generated so far: {self.tokenizer.decode(hub_tokens)}")
                print()
            gc.collect()

        print("-------- Response Summary --------")
        print(f"Prompt: {input_prompt}")
        print(f"Response: {self.tokenizer.decode(hub_tokens)}")
