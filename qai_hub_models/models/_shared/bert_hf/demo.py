# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

from qai_hub_models.models._shared.bert_hf.app import BaseBertApp
from qai_hub_models.models._shared.bert_hf.model import MODEL_ID
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.base_model import BaseModel

INPUT_TEXT = "Paris is the [MASK] of France."


def bert_demo(
    model_type: type[BaseModel],
    model_id: str = MODEL_ID,
    default_text: str = INPUT_TEXT,
    is_test: bool = False,
):
    """Demo for BERT model fill-mask task."""
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)
    parser.add_argument(
        "--text",
        type=str,
        default=default_text,
        help="Input text with [MASK] to process",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=model_type.get_input_spec()["input_tokens"][0][1],
        help="Maximum sequence length for tokenization",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load model and create app
    model = demo_model_from_cli_args(model_type, model_id, args)
    tokenizer = model_type.from_pretrained().tokenizer
    app = BaseBertApp(model, tokenizer, max_seq_length=args.max_seq_length)  # type: ignore[arg-type]
    results = app.fill_mask(args.text)
    if not is_test:
        print(f"Input: {args.text}")
        print(f"Sequence: {results}")
