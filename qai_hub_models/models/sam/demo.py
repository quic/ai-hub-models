# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse

import numpy as np

from qai_hub_models.models.sam.app import SAMApp
from qai_hub_models.models.sam.model import (
    DEFAULT_MODEL_TYPE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SMALL_MODEL_TYPE,
    SAMQAIHMWrapper,
)
from qai_hub_models.models.sam.utils import show_image
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "truck.jpg"
)


# Run SAM end-to-end model on given image.
# The demo will output image with segmentation mask applied for input points
def main(is_test: bool = False):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        help=f"SAM model type to load. Tested with model type `{DEFAULT_MODEL_TYPE}`.",
    )
    parser.add_argument(
        "--point-coordinates",
        type=str,
        default="500,375;",
        help="Comma separated x and y coordinate. Multiple coordinate separated by `;`."
        " e.g. `x1,y1;x2,y2`. Default: `500,375;`",
    )
    parser.add_argument(
        "--single-mask-mode",
        type=bool,
        default=True,
        help="If True, returns single mask. For multiple points multiple masks could lead to better results.",
    )
    args = parser.parse_args(["--model-type", SMALL_MODEL_TYPE] if is_test else None)

    coordinates = list(filter(None, args.point_coordinates.split(";")))

    # Load Application
    app = SAMApp(SAMQAIHMWrapper.from_pretrained(model_type=args.model_type))

    # Load Image
    image = load_image(args.image)
    image_data = np.asarray(image)

    # Prepare SAM for decoder for given input image:
    # i.e. run SAM encoder to generate and cache image embeddings
    app.prepare(image_data, single_mask_mode=args.single_mask_mode)

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
    generated_mask, *_ = app.generate_mask_from_points(input_coords, input_labels)

    if not is_test:
        show_image(image, generated_mask)


if __name__ == "__main__":
    main()
