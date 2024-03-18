# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.sinet.app import SINetApp
from qai_hub_models.models.sinet.model import MODEL_ASSET_VERSION, MODEL_ID, SINet
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

INPUT_IMAGE_LOCAL_PATH = "sinet_demo.png"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, INPUT_IMAGE_LOCAL_PATH
)


def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(SINet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL.",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(SINet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # load image and model
    image = load_image(args.image)
    input_image = image.convert("RGB")
    app = SINetApp(model)
    output = app.predict(input_image, False, False)
    if not is_test:
        display_or_save_image(output, args.output_dir, "sinet_demo_output.png")


if __name__ == "__main__":
    main()
