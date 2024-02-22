# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Type

from qai_hub_models.models._shared.deeplab.app import DeepLabV3App
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    get_model_cli_parser,
    model_from_cli_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image


def deeplabv3_demo(
    model_type: Type[BaseModel],
    default_image: str | CachedWebAsset,
    num_classes: int,
    is_test: bool,
):
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL.",
    )
    add_output_dir_arg(parser)
    args = parser.parse_args([] if is_test else None)

    # This DeepLabV3 ResNet 50 demo comes from
    # https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    # load image and model
    image = load_image(args.image)
    input_image = image.convert("RGB")
    app = DeepLabV3App(model_from_cli_args(model_type, args), num_classes=num_classes)
    output = app.predict(input_image, False)
    if not is_test:
        display_or_save_image(output, args.output_dir)
