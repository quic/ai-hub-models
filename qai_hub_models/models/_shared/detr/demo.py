# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Type

from PIL import Image

from qai_hub_models.models._shared.detr.app import DETRApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image


# Run DETR app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
def detr_demo(
    model_cls: Type[BaseModel],
    model_id: str,
    default_weights: str,
    default_image: str | CachedWebAsset,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="test image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load image & model
    detr = demo_model_from_cli_args(model_cls, model_id, args)
    if isinstance(detr, model_cls):
        input_spec = detr.get_input_spec()
    else:
        input_spec = model_cls.get_input_spec()
    (h, w) = input_spec["image"][0][2:]

    # Run app to scores, labels and boxes
    img = load_image(args.image)
    app = DETRApp(detr, h, w)
    pred_images, _, _, _ = app.predict(img, default_weights)
    pred_image = Image.fromarray(pred_images[0])

    # Show the predicted boxes, scores and class names on the image.
    if is_test:
        assert isinstance(pred_image, Image.Image)
    else:
        display_or_save_image(pred_image, args.output_dir)
