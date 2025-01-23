# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.openpose.app import OpenPoseApp
from qai_hub_models.models.openpose.model import IMAGE_ADDRESS, MODEL_ID, OpenPose
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad


# Run OpenPose end-to-end on a sample image.
# The demo will display the input image with circles drawn over the estimated joint positions.
def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(OpenPose)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL.",
    )

    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(OpenPose, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image
    app = OpenPoseApp(model)
    (_, _, height, width) = OpenPose.get_input_spec()["image"][0]
    orig_image = load_image(args.image)
    image, scale, padding = pil_resize_pad(orig_image, (height, width))

    # Run inference
    pred_image = app.estimate_pose(image)

    # Resize / unpad annotated image
    pred_image = pil_undo_resize_pad(pred_image, orig_image.size, scale, padding)

    if not is_test:
        display_or_save_image(pred_image, args.output_dir)


if __name__ == "__main__":
    main()
