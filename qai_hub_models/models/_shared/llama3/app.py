# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import GenerationConfig

from qai_hub_models.models._shared.llama3.model import RopeEmbedding
from qai_hub_models.models._shared.llm.generator import LLM_Generator, LLM_Loader
from qai_hub_models.utils.base_model import BaseModel


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
        self.tokenizer = tokenizer
        self.end_tokens = end_tokens
        self.seed = seed

    def generate_output_prompt(
        self,
        input_prompt: str,
        context_length: int,
        max_output_tokens: int,
        checkpoint: str | None = None,
        model_from_pretrained_extra: dict = {},
    ):
        torch.manual_seed(self.seed)
        input_prompt_processed = self.get_input_prompt_with_tags(
            user_input_prompt=input_prompt
        )

        host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_tokens = self.tokenizer(
            input_prompt_processed,
            return_tensors="pt",
        ).to(host_device)

        model_params = {
            "context_length": context_length,
            "host_device": host_device,
            **model_from_pretrained_extra,
        }
        if checkpoint is not None:
            model_params["checkpoint"] = checkpoint

        models = [
            LLM_Loader(self.model_cls, sequence_length, model_params, host_device)
            for sequence_length in (1, 128)
        ]
        if "fp_model" in model_from_pretrained_extra:
            config = model_from_pretrained_extra["fp_model"].llm_config
        else:
            config = models[-1].load().llm_config

        rope_embedding = RopeEmbedding(max_length=context_length, config=config)
        inferencer = LLM_Generator(models, self.tokenizer, rope_embedding)

        # can set temperature, topK, topP, etc here
        inferencer.generation_config = GenerationConfig(
            max_new_tokens=max_output_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            temperature=0.8,
        )

        output = inferencer.generate(
            inputs=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            generation_config=inferencer.generation_config,
        )

        print("-------- Response Summary --------")
        print(f"Prompt: {input_prompt_processed}")
        prompt_length = input_tokens["input_ids"][0].shape[-1]
        print(f"Response: {self.tokenizer.decode(output[0][prompt_length:])}")
