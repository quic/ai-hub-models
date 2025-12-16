# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.electra_bert_base_discrim_google.app import (
    ElectraBertApp,
)
from qai_hub_models.models.electra_bert_base_discrim_google.model import (
    MODEL_ID,
    ElectraBertBaseDiscrimGoogle,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.base_model import BaseModel

INPUT_TEXT = "The quick brown fox fake over the lazy dog"


def bert_demo(
    model_type: type[BaseModel],
    model_id: str = MODEL_ID,
    default_text: str = INPUT_TEXT,
    is_test: bool = False,
    max_seq_length: int = 384,
):
    """Demo for BERT model text replacement task."""
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)
    parser.add_argument(
        "--text",
        type=str,
        default=default_text,
        help="Input text to process for replacements",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=max_seq_length,
        help="Maximum sequence length for tokenization",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load model and create app
    model = demo_model_from_cli_args(model_type, model_id, args)
    tokenizer = model_type.from_pretrained().tokenizer
    app = ElectraBertApp(model, tokenizer, max_seq_length=args.max_seq_length)  # type: ignore[arg-type]

    results = app.detect_replacements(args.text)
    if not is_test:
        print(f"Input: {args.text}")
        print(f"Sequence: {results}")


def main(is_test: bool = False):
    bert_demo(ElectraBertBaseDiscrimGoogle, MODEL_ID, INPUT_TEXT, is_test)


if __name__ == "__main__":
    main()
