# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.cavaface.app import CavaFaceApp
from qai_hub_models.models.cavaface.model import MODEL_ASSET_VERSION, MODEL_ID, CavaFace
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel

INPUT_IMAGE_ADDRESS_1 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "cavaface_demo_input_1.jpg"
)
INPUT_IMAGE_ADDRESS_2 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "cavaface_demo_input_2.jpg"
)


def cavaface_demo(
    model_type: type[BaseModel],
    model_id: str,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image1",
        type=str,
        default=INPUT_IMAGE_ADDRESS_1,
        help="image file path or URL",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=INPUT_IMAGE_ADDRESS_2,
        help="second image file path or URL",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="Use both original and flipped images",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load model & image
    model = demo_model_from_cli_args(model_type, model_id, args)
    (height, width) = model_type.get_input_spec()["image"][0][2:]
    image1 = load_image(args.image1)

    app = CavaFaceApp(model, height, width)  # type: ignore[arg-type]
    print("Model loaded")

    emb1 = app.predict_features(image1, use_flip=args.flip)

    if is_test:
        return emb1

    image2 = load_image(args.image2)
    emb2 = app.predict_features(image2, use_flip=args.flip)
    sim_emb = np.dot(emb1, emb2)
    similarity_percentage = (sim_emb) * 100
    is_similar = sim_emb > 0.5
    print(
        f"similarity {similarity_percentage:.2f}% : same person: {'Yes' if is_similar else 'No'}"
    )


def main(is_test: bool = False):
    cavaface_demo(CavaFace, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
