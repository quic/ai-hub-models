# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qai_hub_models.models.openai_clip.app import ClipApp
from qai_hub_models.models.openai_clip.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OpenAIClip,
)
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

DEFAULT_IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]


# Run Clip on a directory of images with a query text.
# The demo will display similarity score for each image.
def main(is_test: bool = False):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-paths",
        type=str,
        default="",
        help="Specify paths of images to load.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="camping under the stars",
        help="Text prompt for image search",
    )
    add_output_dir_arg(parser)
    args = parser.parse_args([] if is_test else None)

    # Load model
    clip_model = OpenAIClip.from_pretrained()
    app = ClipApp(clip_model, clip_model.text_tokenizer, clip_model.image_preprocessor)

    image_paths: list[str | Path] = []
    if args.image_paths:
        image_paths = [x.strip() for x in args.image_paths.split(",")]
    else:
        for file in DEFAULT_IMAGES:
            asset = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, file
            )
            asset.fetch()
            image_paths.append(asset.path())

    predictions = app.predict_similarity(image_paths, [args.text]).flatten()

    # Display all the images and their score wrt to the text prompt provided.
    print(f"Searching images by prompt: {args.text}")
    for i in range(len(predictions)):
        print(
            f"\t Image with name: {image_paths[i]} has a similarity score={predictions[i]}"
        )

    # Show image
    print("Displaying the most relevant image")

    selected_image = image_paths[np.argmax(predictions)]
    most_relevant_image = load_image(selected_image)

    if not is_test:
        display_or_save_image(most_relevant_image, args.output_dir)


if __name__ == "__main__":
    main()
