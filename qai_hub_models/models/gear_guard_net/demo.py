# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Callable

import numpy as np
from PIL import Image

from qai_hub_models.models.gear_guard_net.app import BodyDetectionApp
from qai_hub_models.models.gear_guard_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    GearGuardNet,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_image.jpg"
)


# The demo will display an image with the predicted gear guard bounding boxes.
def gear_guard_demo(
    model_type: type[BaseModel],
    model_id: str,
    app_type: Callable[..., BodyDetectionApp],
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    image_help = "image file path or URL."
    parser.add_argument(
        "--image", type=str, default=INPUT_IMAGE_ADDRESS, help=image_help
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.9,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)

    app = app_type(
        model,
        args.score_threshold,
        args.iou_threshold,
        args.include_postprocessing,
    )

    print("Model Loaded")
    image = load_image(args.image)
    pred_images = app.predict_boxes_from_image(image)
    assert isinstance(pred_images[0], np.ndarray)
    out = Image.fromarray(pred_images[0])
    if not is_test:
        display_or_save_image(out, args.output_dir, "gear_guard_demo_output.png")


def main(is_test: bool = False) -> None:
    gear_guard_demo(
        model_type=GearGuardNet,
        model_id=MODEL_ID,
        app_type=BodyDetectionApp,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
