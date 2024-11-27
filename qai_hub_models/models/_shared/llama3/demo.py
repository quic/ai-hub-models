# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import Any

from qai_hub_models.models._shared.llama3.app import ChatApp as App
from qai_hub_models.utils.args import get_model_cli_parser
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
MAX_OUTPUT_TOKENS = 20


def llama_chat_demo(
    model_cls: type[BaseModel],
    model_id: str,
    get_input_prompt_with_tags: Callable,
    prepare_combined_attention_mask: Callable,
    tokenizer: Any,
    end_tokens: set[str],
    hf_repo_name: str,
    hf_repo_url: str,
    default_prompt: str,
    is_test: bool = False,
    available_target_runtimes: list[TargetRuntime] = [TargetRuntime.QNN],
    bundled_kvcache: bool = True,
):
    """
    Shared Chat Demo App to generate output for provided input prompt
        model_cls: Model base class (either Prompt Processor or Token Generator)
        model_id: Model ID from hub,
        get_input_prompt_with_tags: Function to wrap input prompt with appropriate tags,
        prepare_combined_attention_mask: Function to combine attention mask,
        tokenizer: Tokenizer to encode-decode prompt,
        num_splits: Number of model splits,
        end_tokens: Set of end tokens to use for end of output generation,
        hf_repo_name: HF repo name,
        hf_repo_url: HF repo url,
        default_prompt: Default prompt to set,
        is_test: If test, no options required,
        available_target_runtimes: Default availble runtime in options,
        bundled_kvcache: KV-cache for each head is concatenated.
    """
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="input prompt.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help="max output tokens to generate.",
    )
    args = parser.parse_args([] if is_test else None)

    if not is_test:
        print(f"\n{'-' * 85}")
        print(f"** Generating response via {model_id} **")
        print()
        print("Prompt:", args.prompt)
        print("Max number of output tokens to generate:", args.max_output_tokens)
        print("Please pass `--max-output-tokens <int>` to generate longer responses.")
        print()
        print(
            textwrap.dedent(
                """
            NOTE: This demo runs an unquantized version of Llama, so it may
                  not be representative of on-device results. The demo is intended as
                  reference code for how Llama can be executed on device using both a
                  prompt processor and a token generator. We recommend using Genie
                  SDK for on-device deployment of LLMs.""".lstrip(
                    "\n"
                )
            )
        )
        print(f"{'-' * 85}\n")

    has_model_access(hf_repo_name, hf_repo_url)

    app = App(
        model_cls,
        get_input_prompt_with_tags=get_input_prompt_with_tags,
        prepare_combined_attention_mask=prepare_combined_attention_mask,
        tokenizer=tokenizer,
        end_tokens=end_tokens,
    )
    app.generate_output_prompt(
        args.prompt,
        prompt_sequence_length=args.sequence_length,
        context_length=args.context_length,
        max_output_tokens=args.max_output_tokens,
        bundled_kvcache=bundled_kvcache,
    )
