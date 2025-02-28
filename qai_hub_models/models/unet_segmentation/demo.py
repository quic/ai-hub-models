# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Callable

import torch
from PIL import Image
from PIL.Image import fromarray

from qai_hub_models.models.unet_segmentation.app import UNetSegmentationApp
from qai_hub_models.models.unet_segmentation.model import IMAGE_ADDRESS, MODEL_ID, UNet
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import PathType, load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.image_processing import pil_resize_pad


# Run unet segmentation app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
def unet_demo(
    model: Callable[..., Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    MODEL_ID,
    default_image: PathType,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(UNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="File path or URL to an input image to use for the demo.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image & model
    # demo_model_from_cli_args might return an OnDeviceModel
    model = demo_model_from_cli_args(UNet, MODEL_ID, args)  # type: ignore[assignment]
    print("Model loaded from pre-trained weights.")
    (_, _, height, width) = UNet.get_input_spec()["image"][0]
    orig_image = load_image(
        args.image or default_image, verbose=True, desc="sample input image"
    )
    image, _, _ = pil_resize_pad(orig_image, (height, width))

    # Run app
    app = UNetSegmentationApp(model)
    mask = fromarray(app.predict(image))
    if not is_test:
        output = Image.blend(image.convert("RGBA"), mask.convert("RGBA"), alpha=0.5)
        display_or_save_image(output, args.output_dir, "output.png", "output")


def main(is_test: bool = False):
    unet_demo(
        UNet,
        MODEL_ID,
        IMAGE_ADDRESS,
        is_test,
    )


if __name__ == "__main__":
    main()
