# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from copy import deepcopy

import numpy as np
import PIL.Image as Image

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
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.draw import draw_box_from_corners

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_image.jpg"
)


def plot_result(img: np.ndarray, result: np.ndarray):
    """
    Plot detection result.

    Inputs:
        img: np.ndarray
            Input image.
        result: np.ndarray
            Detection result.
    """
    box_color = ((255, 0, 0), (0, 255, 0))
    for r in result:
        corners = np.array(
            [[r[1], r[2]], [r[1], r[4]], [r[3], r[2]], [r[3], r[4]]]
        ).astype(int)
        draw_box_from_corners(img, corners, box_color[int(r[0])])
    return img


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(GearGuardNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Detection confidence",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(GearGuardNet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    (_, _, height, width) = model.get_input_spec()["image"][0]
    app = BodyDetectionApp(model)  # type: ignore[arg-type]
    result = app.detect(args.image, height, width, args.confidence)

    if not is_test:
        img = np.array(load_image(args.image))
        image_annotated = plot_result(deepcopy(img), result)
        display_or_save_image(
            Image.fromarray(image_annotated), args.output_dir, "result.jpg"
        )


if __name__ == "__main__":
    main()
