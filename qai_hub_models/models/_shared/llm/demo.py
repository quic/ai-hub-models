# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

from qai_hub_models.models._shared.llm.app import ChatApp as App
from qai_hub_models.models._shared.llm.model import (
    LLM_AIMETOnnx,
    LLMBase,
    get_tokenizer,
    is_quantized_checkpoint,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.args import get_model_cli_parser
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
MAX_OUTPUT_TOKENS = 1000


def llm_chat_demo(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    model_id: str,
    prepare_combined_attention_mask: Callable,
    end_tokens: set[str],
    hf_repo_name: str,
    hf_repo_url: str,
    default_prompt: str,
    supported_precisions: list[Precision],
    test_checkpoint: CheckpointSpec | None = None,
    available_target_runtimes: list[TargetRuntime] = [TargetRuntime.QNN_CONTEXT_BINARY],
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
    parser = get_model_cli_parser(
        model_cls,
        suppress_help_arguments=["--host-device", "--fp-model", "--precision"],
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

    args = parser.parse_args([] if test_checkpoint is not None else None)
    checkpoint = args.checkpoint if test_checkpoint is None else test_checkpoint
    max_output_tokens = args.max_output_tokens if test_checkpoint is None else 10
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
        preprocess_prompt_fn = fp_model_cls.get_input_prompt_with_tags

    assert checkpoint is not None
    is_quantized = is_quantized_checkpoint(checkpoint)
    if checkpoint not in {"DEFAULT", "DEFAULT_UNQUANTIZED"}:
        tokenizer = get_tokenizer(checkpoint)
    else:
        has_model_access(hf_repo_name, hf_repo_url)
        tokenizer = get_tokenizer(hf_repo_name)

    if test_checkpoint is None:
        print(f"\n{'-' * 85}")
        print(f"** Generating response via {model_id} **")
        if is_quantized:
            print("Variant: QUANTIZED (AIMET-ONNX)")
            print("    This aims to replicate on-device accuracy through simulation.")
        else:
            print("Variant: FLOATING POINT (PyTorch)")
            print("    This runs the original unquantized model for baseline purposes.")
        print()
        print("Prompt:", prompt)
        print("Raw (prompt will be passed in unchanged):", args.raw)
        print("Max number of output tokens to generate:", args.max_output_tokens)
        print()
        print(f"{'-' * 85}\n")

    extra = {}

    if is_quantized:
        if checkpoint in {"DEFAULT", "DEFAULT_QUANTIZED"}:
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
        tokenizer=tokenizer,
        end_tokens=end_tokens,
        seed=args.seed,
    )

    app.generate_output_prompt(
        prompt,
        context_length=args.context_length,
        max_output_tokens=max_output_tokens,
        checkpoint=checkpoint,
        model_from_pretrained_extra=extra,
    )
