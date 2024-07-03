# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Callable, Dict, List, Set, Tuple, Type

import qai_hub as hub

from qai_hub_models.models._shared.llama.app import ChatApp as App
from qai_hub_models.models._shared.llama.app import (
    LlamaModelPipeline,
    OnDeviceLlamaModelPipeline,
)
from qai_hub_models.models._shared.llama.model import DEFAULT_INPUT_SEQ_LEN
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
# Keeping this short as on-device demo takes time to converge.
MAX_OUTPUT_TOKENS = 10
DEFAULT_DEVICE = "Samsung Galaxy S24 (Family)"
DEFAULT_USER_PROMPT = "Hi! What is 2+3?"


def llama_chat_demo(
    model_cls: Type[BaseModel],
    model_id: str,
    get_model_class: Callable,
    get_input_prompt_with_tags: Callable,
    prepare_combined_attention_mask: Callable,
    tokenizer: Any,
    num_splits: int,
    num_key_val_heads: int,
    model_split_map: Dict[int, Tuple[int, int]],
    end_tokens: Set[str],
    hf_repo_name: str,
    hf_repo_url: str,
    default_prompt: str = DEFAULT_USER_PROMPT,
    is_test: bool = False,
    available_target_runtimes: List[TargetRuntime] = [TargetRuntime.QNN],
):
    """
    Shared Chat Demo App to generate output for provided input prompt
        model_cls: Model base class (either Prompt Processor or Token Generator)
        model_id: Model ID from hub,
        get_model_class: Function to get initialize model class,
        get_input_prompt_with_tags: Function to wrap input prompt with appropriate tags,
        prepare_combined_attention_mask: Function to combine attention mask,
        tokenizer: Tokenizer to encode-decode prompt,
        num_splits: Number of model splits,
        num_key_val_heads: Number of heads in past key-value cache,
        model_split_map: Map for split number to decoder layer ranges,
        end_tokens: Set of end tokens to use for end of output generation,
        hf_repo_name: HF repo name,
        hf_repo_url: HF repo url,
        default_prompt: Default prompt to set,
        is_test: If test, no options required,
        available_target_runtimes: Default availble runtime in options,
    """
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
        available_target_runtimes=available_target_runtimes,
        default_device=DEFAULT_DEVICE,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="input prompt.",
    )
    parser.add_argument(
        "--prompt-processor-input-seq-len",
        type=int,
        default=DEFAULT_INPUT_SEQ_LEN,
        help="input sequence length for prompt-processor. This must be less than `max_position_embeddings` set for model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help="max output tokens to generate.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

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

    if not args.on_device:
        prompt_processor = LlamaModelPipeline(
            model_cls.from_pretrained(),
            num_splits=num_splits,
            num_past_key_val_heads=num_key_val_heads,
            model_split_map=model_split_map,
        )
        token_generator = LlamaModelPipeline(
            model_cls.from_pretrained(),
            num_splits=num_splits,
            num_past_key_val_heads=num_key_val_heads,
            model_split_map=model_split_map,
            is_token_generator=True,
        )
    else:
        hub_model_ids = args.hub_model_id.split(",")
        # First four models are Prompt Processor
        # Last four models are Token Generator
        if len(hub_model_ids) != num_splits * 2:
            model_id_lists = ",".join(
                [f"<id-{i}>" for i in range(1, num_splits * 2 + 1)]
            )
            raise RuntimeError(
                "Please provide comma separated hub-model-ids for Llama Prompt Processor and Token Generator,"
                f" e.g. --hub-model-id {model_id_lists}.\n"
                "Specify model-ids for four Prompt Processor models first, then Token Generator models.\n"
                "If you run export.py it will print out command to run on-device demo with ordered model-ids."
            )

        hub_device = hub.Device(args.device)
        prompt_processor = OnDeviceLlamaModelPipeline(
            hub_model_ids[:num_splits],
            hub_device=hub_device,
            inference_options=args.inference_options,
            get_model_class=get_model_class,
            num_past_key_val_heads=num_key_val_heads,
            model_split_map=model_split_map,
        )
        token_generator = OnDeviceLlamaModelPipeline(
            hub_model_ids[num_splits:],
            hub_device=hub_device,
            inference_options=args.inference_options,
            get_model_class=get_model_class,
            num_past_key_val_heads=num_key_val_heads,
            model_split_map=model_split_map,
            is_token_generator=True,
        )

    has_model_access(hf_repo_name, hf_repo_url)

    app = App(
        prompt_processor,
        token_generator,
        get_input_prompt_with_tags=get_input_prompt_with_tags,
        prepare_combined_attention_mask=prepare_combined_attention_mask,
        tokenizer=tokenizer,
        end_tokens=end_tokens,
        num_past_key_val_heads=num_key_val_heads,
    )
    app.generate_output_prompt(
        args.prompt,
        max_seq_len=args.prompt_processor_input_seq_len,
        max_output_tokens=args.max_output_tokens,
    )
