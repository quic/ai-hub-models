# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from PIL.Image import Image

from qai_hub_models.models.sinet.app import SINetApp
from qai_hub_models.models.sinet.model import INPUT_IMAGE_ADDRESS, MODEL_ID, SINet
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image


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
    app = SINetApp(model)  # type: ignore[arg-type]
    output = app.predict(input_image, False, False)
    if not is_test:
        assert isinstance(output, Image)
        display_or_save_image(output, args.output_dir, "sinet_demo_output.png")


if __name__ == "__main__":
    main()
