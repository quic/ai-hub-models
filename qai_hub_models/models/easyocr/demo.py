# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.easyocr.app import EasyOCRApp
from qai_hub_models.models.easyocr.model import MODEL_ASSET_VERSION, MODEL_ID, EasyOCR
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "english.png"
)


def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(EasyOCR)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    if not isinstance(args.lang_list, list):
        raise TypeError("The 'lang_list' parameter must be a list.")
    validate_on_device_demo_args(args, MODEL_ID)

    # Load app and image
    image = load_image(args.image)
    model = model_from_cli_args(EasyOCR, args)
    app = EasyOCRApp(model.detector, model.recognizer, model.lang_list)
    print("Model Loaded")

    results = app.predict_text_from_image(image)

    if not is_test:
        display_or_save_image(results[0], args.output_dir)


if __name__ == "__main__":
    main()
