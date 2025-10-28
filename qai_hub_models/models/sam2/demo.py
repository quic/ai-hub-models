# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from pathlib import Path

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from qai_hub_models.models._shared.sam.utils import show_image
from qai_hub_models.models.sam2.app import SAM2App, SAM2InputImageLayout
from qai_hub_models.models.sam2.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAM2,
    TINY_MODEL_TYPE,
)
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.evaluate import EvalMode

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

    Parameters
    ----------
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
    add_output_dir_arg(parser)
    get_on_device_demo_parser(parser)

    args = parser.parse_args(["--model-type", TINY_MODEL_TYPE] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    coordinates: list[str] = list(filter(None, args.point_coordinates.split(";")))
    # Load Application
    wrapper = SAM2.from_pretrained()

    if args.eval_mode == EvalMode.ON_DEVICE:
        if args.hub_model_id:
            sam2_encoder, sam2_decoder = demo_model_components_from_cli_args(
                SAM2, MODEL_ID, args
            )
        else:
            raise ValueError(
                "If running this demo with on device, must supply hub_model_id."
            )
    else:
        sam2_encoder = wrapper.encoder
        sam2_decoder = wrapper.decoder

    app = SAM2App(
        encoder_input_img_size=wrapper.encoder.sam2.image_size,
        mask_threshold=SAM2ImagePredictor(wrapper.encoder.sam2).mask_threshold,
        input_image_channel_layout=SAM2InputImageLayout["RGB"],
        sam2_encoder=sam2_encoder,  # type: ignore[arg-type]
        sam2_decoder=sam2_decoder,  # type: ignore[arg-type]
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
        show_image(
            image, np.asarray(generated_mask), input_coords, str(Path.cwd() / "build")
        )


if __name__ == "__main__":
    main()
