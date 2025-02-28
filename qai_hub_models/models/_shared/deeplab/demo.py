# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from PIL import Image

from qai_hub_models.models._shared.deeplab.app import DeepLabV3App
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad


def deeplabv3_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: str | CachedWebAsset,
    num_classes: int,
    is_test: bool,
):
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    input_spec = model_type.get_input_spec()

    # load image and model
    (_, _, height, width) = input_spec["image"][0]
    orig_image = load_image(args.image)
    image, scale, padding = pil_resize_pad(orig_image, (height, width))

    # This DeepLabV3 ResNet 50 demo comes from
    # https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    input_image = image.convert("RGB")
    inference_model = demo_model_from_cli_args(model_type, model_id, args)
    app = DeepLabV3App(inference_model, num_classes=num_classes)

    # Run app
    image_annotated = app.predict(input_image, False)
    assert isinstance(image_annotated, Image.Image)

    # Resize / unpad annotated image
    image_annotated = pil_undo_resize_pad(
        image_annotated, orig_image.size, scale, padding
    )

    if not is_test:
        display_or_save_image(
            image_annotated, args.output_dir, "annotated_image.png", "predicted image"
        )
