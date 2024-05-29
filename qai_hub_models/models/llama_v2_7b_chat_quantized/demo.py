# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List, Type

import qai_hub as hub
from transformers import LlamaTokenizer

from qai_hub_models.models.llama_v2_7b_chat_quantized import MODEL_ID, Model
from qai_hub_models.models.llama_v2_7b_chat_quantized.app import ChatApp as App
from qai_hub_models.models.llama_v2_7b_chat_quantized.app import (
    HubLlama2ModelPipeline,
    Llama2ModelPipeline,
)
from qai_hub_models.models.llama_v2_7b_chat_quantized.model import (
    DEFAULT_INPUT_SEQ_LEN,
    DEFAULT_USER_PROMPT,
    HF_REPO_NAME,
    HF_REPO_URL,
)
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
DEFAULT_DEVICE = "Samsung Galaxy S24"


def llama_chat_demo(
    model_cls: Type[BaseModel] = Model,
    model_id: str = MODEL_ID,
    default_prompt: str = DEFAULT_USER_PROMPT,
    is_test: bool = False,
    available_target_runtimes: List[TargetRuntime] = [TargetRuntime.QNN],
):
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

    if not args.on_device:
        prompt_processor = Llama2ModelPipeline(Model.from_pretrained())
        token_generator = Llama2ModelPipeline(
            Model.from_pretrained(), is_token_generator=True
        )
    else:
        hub_model_ids = args.hub_model_id.split(",")
        # First four models are Prompt Processor
        # Last four models are Token Generator
        if len(hub_model_ids) != 8:
            raise RuntimeError(
                "Please provide comma separated hub-model-ids for Llama Prompt Processor and Token Generator,"
                " e.g. --hub-model-id <id-1>,<id-2>,<id-3>,<id-4>,<id-5>,<id-6>,<id-7>,<id-8>.\n"
                "Specify model-ids for four Prompt Processor models first, then Token Generator models.\n"
                "If you run export.py it will print out command to run on-device demo with ordered model-ids."
            )

        hub_device = hub.Device(args.device)
        prompt_processor = HubLlama2ModelPipeline(
            hub_model_ids[:4],
            hub_device=hub_device,
            inference_options=args.inference_options,
        )
        token_generator = HubLlama2ModelPipeline(
            hub_model_ids[4:],
            hub_device=hub_device,
            inference_options=args.inference_options,
            is_token_generator=True,
        )

    has_model_access(HF_REPO_NAME, HF_REPO_URL)
    tokenizer = LlamaTokenizer.from_pretrained(HF_REPO_NAME)

    app = App(prompt_processor, token_generator, tokenizer=tokenizer)
    app.generate_output_prompt(
        args.prompt,
        max_seq_len=args.prompt_processor_input_seq_len,
        max_output_tokens=args.max_output_tokens,
    )


if __name__ == "__main__":
    llama_chat_demo()
