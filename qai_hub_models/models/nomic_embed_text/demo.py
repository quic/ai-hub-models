# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.nomic_embed_text.app import NomicEmbedTextApp
from qai_hub_models.models.nomic_embed_text.model import MODEL_ID, NomicEmbedText
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)


# The demo will display a image with the predicted keypoints.
def nomic_embed_text_demo(
    model_cls: type[NomicEmbedText],
    is_test: bool = False,
    default_text: str | None = None,
):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--text",
        type=str,
        help="Text to encode",
        default=default_text,
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    print("Model Loaded")

    app = NomicEmbedTextApp(model, args.sequence_length)
    print(f"Embeddings: \n{app.predict(args.text)}")


def main(is_test: bool = False, default_text=None):
    return nomic_embed_text_demo(NomicEmbedText, is_test, default_text)


if __name__ == "__main__":
    main()
