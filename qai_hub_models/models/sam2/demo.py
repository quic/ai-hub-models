# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from pathlib import Path
from typing import cast

import numpy as np
import torch

from qai_hub_models.models._shared.sam.utils import show_image
from qai_hub_models.models.sam2.app import SAM2App, SAM2InputImageLayout
from qai_hub_models.models.sam2.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAM2,
    TINY_MODEL_TYPE,
)
from qai_hub_models.models.sam2.model_patches import SAM2ImagePredictor
from qai_hub_models.utils.args import demo_model_from_cli_args, get_model_cli_parser
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "truck.jpg"
)

# Run SAM end-to-end model on given image.
# The demo will output image with segmentation mask applied for input points


def main(is_test: bool = False) -> None:
    """
    Main function to run a point-based image segmentation demo using the SAM2 model.

    This function parses command-line arguments, loads the SAM2 model and image,
    processes user-provided point coordinates, and performs segmentation using those points.
    The resulting mask is displayed and saved unless running in test mode.

    Args:
        is_test (bool, optional): If True, skips displaying the image and storing results.
                                  Useful for automated testing. Defaults to False.

    Workflow:
        1. Parses CLI arguments for model type, image path, and point coordinates.
        2. Loads the SAM2 model and initializes the segmentation application.
        3. Loads the input image from the URL.
        4. Parses and validates point coordinates for segmentation.
        5. Runs the segmentation model to generate a mask.
        6. Displays and stores the result unless in test mode.
    """

    # Demo parameters
    parser = get_model_cli_parser(SAM2)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL",
    )
    parser.add_argument(
        "--point-coordinates",
        type=str,
        default="500,375;1100,600",
        help="Comma separated x and y coordinate. Multiple coordinate separated by `;`."
        " e.g. `x1,y1;x2,y2`. Default: `500,375;`",
    )

    args = parser.parse_args(["--model-type", TINY_MODEL_TYPE])

    coordinates: list[str] = list(filter(None, args.point_coordinates.split(";")))
    # Load Application
    wrapper = cast(SAM2, demo_model_from_cli_args(SAM2, MODEL_ID, args))

    app = SAM2App(
        encoder_input_img_size=wrapper.encoder.sam2.image_size,
        mask_threshold=SAM2ImagePredictor(wrapper.encoder.sam2).mask_threshold,
        input_image_channel_layout=SAM2InputImageLayout["RGB"],
        sam2_encoder=wrapper.encoder,
        sam2_decoder=wrapper.decoder,
    )
    # Load Image
    image = load_image(args.image)

    # Point segmentation using decoder
    print("\n** Performing point segmentation **\n")

    # Input points
    input_coords = []
    input_labels = []

    for coord in coordinates:
        coord_split = coord.split(",")
        if len(coord_split) != 2:
            raise RuntimeError(
                f"Expecting comma separated x and y coordinate. Provided {coord_split}."
            )

        input_coords.append([int(coord_split[0]), int(coord_split[1])])
        # Set label to `1` to include current point for segmentation
        input_labels.append(1)

    # Generate masks with given input points
    generated_mask, *_ = app.predict_mask_from_points(
        image, torch.Tensor(input_coords), torch.Tensor(input_labels)
    )

    if not is_test:
        show_image(image, np.asarray(generated_mask), str(Path.cwd() / "build"))


if __name__ == "__main__":
    main()
