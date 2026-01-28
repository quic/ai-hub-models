# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import partial

from qai_hub_models.models._shared.llm.app import ChatApp as App
from qai_hub_models.models._shared.llm.model import (
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
    get_tokenizer,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.args import get_model_cli_parser
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
)
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
MAX_OUTPUT_TOKENS = 1000


def llm_chat_demo(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    model_id: str,
    end_tokens: set[str],
    hf_repo_name: str,
    hf_repo_url: str,
    supported_precisions: list[Precision],
    default_prompt: str | None = None,
    raw: bool = False,
    test_checkpoint: CheckpointSpec | None = None,
    supports_thinking: bool = False,
) -> None:
    """Shared Chat Demo App to generate output for provided input prompt"""
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
    if supports_thinking:
        parser.add_argument(
            "--thinking",
            action="store_true",
            dest="thinking",
            default=True,
            help="Enable thinking mode (default).",
        )
        parser.add_argument(
            "--no-thinking",
            action="store_false",
            dest="thinking",
            help="Disable thinking mode by adding empty thinking tags.",
        )

    args = parser.parse_args([] if test_checkpoint is not None else None)
    checkpoint = args.checkpoint if test_checkpoint is None else test_checkpoint
    max_output_tokens = args.max_output_tokens if test_checkpoint is None else 1000
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Must specify one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    elif default_prompt is not None:
        prompt = default_prompt
    else:
        prompt = fp_model_cls.default_user_prompt

    # Make sure that we can pass "\n" (0x0A) as part of the prompt, since that
    # is often a common feature of prompt formats. If this gets interpreted as
    # "\\n" (0x5C 0x6E), the LLM can react poorly (quantized models have been
    # observed to be particularly sensitive to this).
    prompt = prompt.replace("\\n", "\n")

    assert checkpoint is not None
    checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
    is_default = isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
    if not is_default:
        tokenizer = get_tokenizer(checkpoint)
    else:
        has_model_access(hf_repo_name, hf_repo_url)
        tokenizer = get_tokenizer(hf_repo_name)

    # Build the prompt formatting function
    if args.raw or raw:

        def preprocess_prompt_fn(
            user_input_prompt: str = "",
            system_context_prompt: str = "",
        ) -> str:
            return user_input_prompt
    elif supports_thinking:
        preprocess_prompt_fn = partial(
            fp_model_cls.get_input_prompt_with_tags,
            tokenizer=tokenizer,
            enable_thinking=args.thinking,
        )
    else:
        preprocess_prompt_fn = partial(
            fp_model_cls.get_input_prompt_with_tags, tokenizer=tokenizer
        )

    if test_checkpoint is None:
        print(f"\n{'-' * 85}")
        print(f"** Generating response via {model_id} **")
        if checkpoint_type == CheckpointType.GENIE_BUNDLE:
            print("Variant: ON-DEVICE (QNN)")
            print("    This runs on the target hardware.")
        elif checkpoint_type.is_aimet_onnx():
            print("Variant: QUANTIZED (AIMET-ONNX)")
            print("    This aims to replicate on-device accuracy through simulation.")
        else:
            print("Variant: FLOATING POINT (PyTorch)")
            print("    This runs the original unquantized model for baseline purposes.")
        print()
        print("Prompt:", prompt)
        print("Raw (prompt will be passed in unchanged):", args.raw)
        if supports_thinking:
            print("Thinking mode:", "enabled" if args.thinking else "disabled")
        print("Max number of output tokens to generate:", args.max_output_tokens)
        print()
        print(f"{'-' * 85}\n")

    extra = {}

    final_model_cls: type[LLMBase | LLM_AIMETOnnx | LLM_QNN]

    if checkpoint_type == CheckpointType.GENIE_BUNDLE:
        final_model_cls = qnn_model_cls

    elif checkpoint_type.is_aimet_onnx():
        if is_default and checkpoint != "DEFAULT_UNQUANTIZED":
            extra["fp_model"] = fp_model_cls.from_pretrained(
                sequence_length=args.sequence_length,
                context_length=args.context_length,
            )
        final_model_cls = model_cls
    else:
        final_model_cls = fp_model_cls

    app = App(
        final_model_cls,
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
