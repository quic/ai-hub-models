# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from typing import Type

from PIL import Image

from qai_hub_models.models._shared.fastsam.app import FastSAMApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebAsset,
    load_image,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image


def fastsam_demo(
    model_type: Type[BaseModel],
    model_id: str,
    image_path: str | CachedWebAsset,
    is_test: bool,
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
    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)
    app = FastSAMApp(model)

    image = load_image(args.image)

    with qaihm_temp_dir() as tmpdir:
        image_path = os.path.join(tmpdir, "inp_image.jpg")
        image.save(image_path)
        pred, prompt_process = app.segment_image(image_path)

        # Store the output image
        output_path = os.path.join(args.output_dir or tmpdir, "output.jpg")

        # Save the output
        prompt_process.plot(annotations=pred, output=output_path)

        if is_test:
            assert pred is not None
        else:
            display_or_save_image(
                Image.open(output_path), args.output_dir, "output.jpg"
            )
