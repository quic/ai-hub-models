# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from collections.abc import Callable

from qai_hub_models.models._shared.llama3_ao.app import ChatApp as App
from qai_hub_models.models._shared.llama3_ao.model import (
    determine_mode,
    verify_mode_and_checkpoint_match,
)
from qai_hub_models.models._shared.llm.model import get_tokenizer
from qai_hub_models.utils.args import get_model_cli_parser
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
MAX_OUTPUT_TOKENS = 1000


def llama_chat_demo(
    model_cls: type[BaseModel],
    fp_model_cls: type[BaseModel],
    model_id: str,
    get_input_prompt_with_tags: Callable,
    prepare_combined_attention_mask: Callable,
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
        model_cls: Model class (of quantized models)
        fp_model_cls: Model class (of floating point models)
        model_id: Model ID from hub,
        get_input_prompt_with_tags: Function to wrap input prompt with appropriate tags,
        prepare_combined_attention_mask: Function to combine attention mask,
        end_tokens: Set of end tokens to use for end of output generation,
        hf_repo_name: HF repo name,
        hf_repo_url: HF repo url,
        default_prompt: Default prompt to set,
        is_test: If test, no options required,
        available_target_runtimes: Available runtimes,
        bundled_kvcache: KV-cache for each head is concatenated.
    """
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fp", "quantsim"],
        help="Run the floating point model or simulated quantization.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="input prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="input prompt from file path.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="If specified, will assume prompt contains systems tags and will not be added automatically.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help="max output tokens to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed.",
    )
    args = parser.parse_args([] if is_test else None)
    mode = args.mode
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Must specify one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = default_prompt

    if args.raw:

        def preprocess_prompt_fn(
            user_input_prompt: str = "",
            system_context_prompt: str = "",
        ):
            return user_input_prompt

    else:
        preprocess_prompt_fn = get_input_prompt_with_tags

    user_specified_checkpoint = "--checkpoint" in sys.argv
    if (not user_specified_checkpoint or args.checkpoint == "DEFAULT") and mode is None:
        parser.error("--mode must be specified if --checkpoint is not.")

    if args.checkpoint != "DEFAULT":
        tokenizer = get_tokenizer(args.checkpoint)

        if not mode:
            mode = determine_mode(args.checkpoint)
        else:
            verify_mode_and_checkpoint_match(args.checkpoint, mode)
    else:
        has_model_access(hf_repo_name, hf_repo_url)
        tokenizer = get_tokenizer(hf_repo_name)

    if not is_test:
        print(f"\n{'-' * 85}")
        print(f"** Generating response via {model_id} **")
        if mode == "fp":
            print("Variant: FP32 (PyTorch)")
            print("    This runs the original unquantized model for baseline purposes.")
        else:
            print("Variant: QUANTIZED (AIMET-ONNX)")
            print("    This aims to replicate on-device accuracy through simulation.")
        print()
        print("Prompt:", prompt)
        print("Raw (prompt will be passed in unchanged):", args.raw)
        print("Max number of output tokens to generate:", args.max_output_tokens)
        print()
        print(f"{'-' * 85}\n")

    extra = {}
    if mode == "quantsim":
        if args.checkpoint == "DEFAULT":
            extra["fp_model"] = fp_model_cls.from_pretrained(
                sequence_length=args.sequence_length,
                context_length=args.context_length,
            )
        model = model_cls
    else:
        model = fp_model_cls

    app = App(
        model,
        get_input_prompt_with_tags=preprocess_prompt_fn,
        prepare_combined_attention_mask=prepare_combined_attention_mask,
        tokenizer=tokenizer,
        end_tokens=end_tokens,
        seed=args.seed,
    )

    app.generate_output_prompt(
        prompt,
        prompt_sequence_length=args.sequence_length,
        context_length=args.context_length,
        max_output_tokens=args.max_output_tokens,
        checkpoint=None
        if args.checkpoint == "DEFAULT" and mode == "fp"
        else args.checkpoint,
        bundled_kvcache=bundled_kvcache,
        model_from_pretrained_extra=extra,
    )
