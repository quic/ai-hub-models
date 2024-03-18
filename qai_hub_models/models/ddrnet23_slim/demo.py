# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.ddrnet23_slim.app import DDRNetApp
from qai_hub_models.models.ddrnet23_slim.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DDRNet,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.png"
)


# Run DDRNet end-to-end on a sample image.
# The demo will display a image with the predicted segmentation map overlaid.
def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(DDRNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(DDRNet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image
    (_, _, height, width) = DDRNet.get_input_spec()["image"][0]
    orig_image = load_image(args.image)
    image, scale, padding = pil_resize_pad(orig_image, (height, width))
    print("Model Loaded")

    app = DDRNetApp(model)
    output = app.segment_image(image)[0]

    if not is_test:
        # Resize / unpad annotated image
        image_annotated = pil_undo_resize_pad(output, orig_image.size, scale, padding)
        display_or_save_image(
            image_annotated, args.output_dir, "ddrnet_demo_output.png"
        )


if __name__ == "__main__":
    main()
