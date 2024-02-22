# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Type

from PIL.Image import fromarray

from qai_hub_models.models.mediapipe_selfie.app import SelfieSegmentationApp
from qai_hub_models.models.mediapipe_selfie.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SelfieSegmentation,
)
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    get_model_cli_parser,
    model_from_cli_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "selfie.jpg"
)


# Run selfie segmentation app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
def mediapipe_selfie_demo(
    model_cls: Type[BaseModel],
    default_image: str | CachedWebModelAsset,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="File path or URL to an input image to use for the demo.",
    )
    add_output_dir_arg(parser)
    args = parser.parse_args([] if is_test else None)

    # Load image & model
    model = model_from_cli_args(model_cls, args)
    print("Model loaded from pre-trained weights.")
    image = load_image(args.image, verbose=True, desc="sample input image")

    # Run app
    app = SelfieSegmentationApp(model)
    mask = app.predict(image) * 255.0
    mask = fromarray(mask).convert("L")
    if not is_test:
        # Make sure the input image and mask are resized so the demo can visually
        # show the images in the same resolution.
        image = image.resize(mask.size)
        display_or_save_image(
            image, args.output_dir, "mediapipe_selfie_image.png", "sample input image"
        )
        display_or_save_image(
            mask, args.output_dir, "mediapipe_selfie_mask.png", "predicted mask"
        )


def main(is_test: bool = False):
    mediapipe_selfie_demo(
        SelfieSegmentation,
        IMAGE_ADDRESS,
        is_test,
    )


if __name__ == "__main__":
    main()
