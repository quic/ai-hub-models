# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.segmentation.app import SegmentationApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad


def segmentation_demo(
    model_type: type[BaseModel],
    model_id,
    default_image: CachedWebModelAsset,
    is_test: bool = False,
    normalize_input: bool = True,
    pad_mode: str = "constant",
    app_cls: type[SegmentationApp] = SegmentationApp,
):
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_type, model_id, args)
    validate_on_device_demo_args(args, model_id)

    (_, _, height, width) = model_type.get_input_spec()["image"][0]
    orig_image = load_image(args.image)
    image, scale, padding = pil_resize_pad(
        orig_image, (height, width), pad_mode=pad_mode
    )

    app = app_cls(model, normalize_input)
    print("Model Loaded")

    output = app.segment_image(image)[0]

    if not is_test:
        # Resize / unpad annotated image
        image_annotated = pil_undo_resize_pad(output, orig_image.size, scale, padding)
        display_or_save_image(image_annotated, args.output_dir)
