# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from PIL.Image import fromarray

from qai_hub_models.models.mediapipe_selfie.app import SelfieSegmentationApp
from qai_hub_models.models.mediapipe_selfie.model import (
    IMAGE_ADDRESS,
    MODEL_ID,
    SelfieSegmentation,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad


# Run selfie segmentation app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
def main(
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(SelfieSegmentation)
    parser = get_on_device_demo_parser(
        parser, available_target_runtimes=[TargetRuntime.TFLITE], add_output_dir=True
    )
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="File path or URL to an input image to use for the demo.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image & model
    orig_image = load_image(args.image)
    model = demo_model_from_cli_args(SelfieSegmentation, MODEL_ID, args)

    # Run app
    # OnDeviceModel is underspecified to meet the Callable type requirements of the following
    app = SelfieSegmentationApp(model)  # type: ignore[reportArgumentType]
    (_, _, height, width) = SelfieSegmentation.get_input_spec()["image"][0]

    image, scale, padding = pil_resize_pad(orig_image, (height, width))
    mask_arr = app.predict(image) * 255.0
    mask = fromarray(mask_arr).convert("L")
    if not is_test:
        # Make sure the input image and mask are resized so the demo can visually
        # show the images in the same resolution.
        image = pil_undo_resize_pad(image, orig_image.size, scale, padding)
        display_or_save_image(
            image, args.output_dir, "mediapipe_selfie_image.png", "sample input image"
        )
        display_or_save_image(
            mask, args.output_dir, "mediapipe_selfie_mask.png", "predicted mask"
        )


if __name__ == "__main__":
    main(is_test=False)
