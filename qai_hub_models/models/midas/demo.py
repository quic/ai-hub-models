# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Type

from qai_hub_models.models.midas.app import MidasApp
from qai_hub_models.models.midas.model import MODEL_ASSET_VERSION, MODEL_ID, Midas
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

# Demo image comes from https://github.com/pytorch/hub/raw/master/images/dog.jpg
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.jpg"
)


# Run Midas end-to-end on a sample image.
# The demo will display a heatmap of the estimated depth at each point in the image.
def midas_demo(model_cls: Type[Midas], is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image
    (_, _, height, width) = model_cls.get_input_spec()["image"][0]
    image = load_image(args.image)
    print("Model Loaded")

    app = MidasApp(model, height, width)
    heatmap_image = app.estimate_depth(image)

    if not is_test:
        # Resize / unpad annotated image
        display_or_save_image(
            heatmap_image, args.output_dir, "midas_heatmap.png", "heatmap"
        )


def main(is_test: bool = False):
    return midas_demo(model_cls=Midas, is_test=is_test)


if __name__ == "__main__":
    main()
