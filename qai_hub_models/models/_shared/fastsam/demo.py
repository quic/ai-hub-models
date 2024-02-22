# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import tempfile
from typing import Type

from PIL import Image

from qai_hub_models.models._shared.fastsam.app import FastSAMApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_path
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image


def fastsam_demo(
    model_type: Type[BaseModel], image_path: str | CachedWebAsset, is_test: bool
):
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=image_path,
        help="image file path or URL.",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_type.get_model_id())

    model = demo_model_from_cli_args(model_type, args)
    app = FastSAMApp(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = load_path(args.image, tmpdir)
        pred, prompt_process = app.segment_image(image_path)

    # Store the output image
    output_dirname, _ = os.path.split(image_path)
    output_path = os.path.join(output_dirname, "output.jpg")
    prompt_process.plot(annotations=pred, output=output_path)

    # Display the output
    output_image = Image.open(output_path)
    if not is_test:
        display_or_save_image(output_image, args.output_dir)
