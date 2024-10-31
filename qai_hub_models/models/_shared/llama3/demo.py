# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qai_hub_models.models._shared.llama3.app import ChatApp as App
from qai_hub_models.utils.args import get_model_cli_parser
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
# Keeping this short as on-device demo takes time to converge.
MAX_OUTPUT_TOKENS = 20
DEFAULT_DEVICE = "Samsung Galaxy S24 (Family)"


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
        "--prompt-processor-input-seq-len",
        type=int,
        default=128,
        help="input sequence length for prompt-processor. This must be less than `context_length` set for model.",
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
            """NOTE: Each token generation takes around 15 mins on-device:
    1. Model is divided into multiple parts to fit into device constraints
    2. Each model requires separate execution on-device via AI Hub
    3. Due to autoregressive nature, we cannot run step 2 in parallel
    4. Device procurement is subject to device availability and might take longer to run demo on-device

Alternative:
    1. Run demo on host (with PyTorch) to verify e2e result for longer responses
    2. Run demo on-device for shorter responses (--max-output-tokens 10 or 20)
    3. [Optional] Can run demo on-device to generate long sentence (takes longer)

We are actively working on to improve UX and reduce turn-around time for these models.
"""
        )
        print(f"{'-' * 85}\n")

    has_model_access(hf_repo_name, hf_repo_url)

    """
    llama_ar128 = model_cls.from_pretrained(
        sequence_length=args.prompt_processor_input_seq_len
    )
    llama_ar1 = model_cls.from_pretrained(sequence_length=1)
    context_length = llama_ar128.context_length
    """

    app = App(
        model_cls,
        get_input_prompt_with_tags=get_input_prompt_with_tags,
        prepare_combined_attention_mask=prepare_combined_attention_mask,
        tokenizer=tokenizer,
        end_tokens=end_tokens,
    )
    context_length = 4096
    app.generate_output_prompt(
        args.prompt,
        prompt_sequence_length=args.prompt_processor_input_seq_len,
        context_length=context_length,
        max_output_tokens=args.max_output_tokens,
        bundled_kvcache=bundled_kvcache,
    )
