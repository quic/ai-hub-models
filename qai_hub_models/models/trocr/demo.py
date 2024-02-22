# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import time

from qai_hub_models.models.trocr.app import TrOCRApp
from qai_hub_models.models.trocr.model import (
    HUGGINGFACE_TROCR_MODEL,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    TrOCR,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

HUGGINGFACE_TROCR_MODEL = "microsoft/trocr-small-stage1"
DEFAULT_SAMPLE_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_text.jpg"
)


# Run TrOCR end-to-end on a sample line of handwriting.
# The demo will output the text contained within the source image.
# Text will be printed to terminal as it is generated with each decoder loop.
def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(TrOCR)
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_SAMPLE_IMAGE,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)

    # Load Application
    app = TrOCRApp(model_from_cli_args(TrOCR, args))

    # Load Image
    image = load_image(args.image)

    # Stream output from model
    print("\n** Predicted Text **\n")

    for output in app.stream_predicted_text_from_image(image):
        if is_test:
            continue
        print(output[0], end="\r")
        # Sleep to accentuate the "streaming" affect in terminal output.
        time.sleep(0.1)

    print("\n")


if __name__ == "__main__":
    main()
