# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse

from qai_hub_models.models.openpose.app import OpenPoseApp
from qai_hub_models.models.openpose.model import MODEL_ASSET_VERSION, MODEL_ID, OpenPose
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "openpose_demo.png"
)


# Run OpenPose end-to-end on a sample image.
# The demo will display the input image with circles drawn over the estimated joint positions.
def main(is_test: bool = False):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL.",
    )
    add_output_dir_arg(parser)

    args = parser.parse_args([] if is_test else None)

    # Load image & model
    app = OpenPoseApp(OpenPose.from_pretrained())
    image = load_image(args.image)
    pred_image = app.estimate_pose(image)
    if not is_test:
        display_or_save_image(pred_image, args.output_dir)


if __name__ == "__main__":
    main()
