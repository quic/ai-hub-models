# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from copy import deepcopy

import numpy as np
import PIL.Image as Image

from qai_hub_models.models._shared.body_detection.app import BodyDetectionApp
from qai_hub_models.models.protocols import FromPretrainedTypeVar
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.draw import draw_box_from_corners


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


def BodyDetectionDemo(
    is_test: bool,
    model_name: type[FromPretrainedTypeVar],
    model_id: str,
    app_name: type[BodyDetectionApp],
    imgfile: str,
    height: int,
    width: int,
    conf: float,
) -> None:
    """
    Object detection demo.

    Input:
        is_test: bool.
            Is test
        model_name: nn.Module
            Object detection model.
        model_id: str.
            Model ID
        app_name: BodyDetectionApp
            Object detection app.
        imgfile: str:
            Image file path.
        height: int
            Input image height.
        width: int
            Input image width.
        conf: float
            Detection confidence.
    """
    parser = get_model_cli_parser(model_name)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=imgfile,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_name, model_id, args)
    validate_on_device_demo_args(args, model_id)

    app = app_name(model)  # type: ignore[arg-type]
    result = app.detect(args.image, height, width, conf)

    if not is_test:
        img = np.array(load_image(args.image))
        image_annotated = plot_result(deepcopy(img), result)
        display_or_save_image(
            Image.fromarray(image_annotated), args.output_dir, "result.jpg"
        )
