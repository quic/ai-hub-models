# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import os

import numpy as np
import torch

from qai_hub_models.models.openai_clip.app import ClipApp
from qai_hub_models.models.openai_clip.model import MODEL_ASSET_VERSION, MODEL_ID, Clip
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image


# Run Clip on a directory of images with a query text.
# The demo will display similarity score for each image.
def main(is_test: bool = False):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to image directory",
    )
    parser.add_argument(
        "--image-names",
        type=str,
        default="image1.jpg,image2.jpg,image3.jpg",
        help="Specify names of the images in the folder.",
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
    clip_model = Clip.from_pretrained()
    app = ClipApp(clip_model=clip_model)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess text
    text = app.process_text(args.text).to(device)

    # Iterate through images and text provided by user
    images = []
    image_names = args.image_names.split(",")
    for filename in image_names:
        # Make sure the file is an image
        if os.path.splitext(filename)[1].lower() in [".jpg", ".jpeg", ".png"]:
            if not args.image_dir:
                image = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, filename
                )
            else:
                image = os.path.join(args.image_dir, filename)
            # Preprocess image and text pair
            image = app.process_image(load_image(image)).to(device)
            images.append(image)

        else:
            print(f"Skipping file {filename}")

    images = torch.stack(images).squeeze(1)

    # Compute similarity
    predictions = app.predict_similarity(images, text).flatten()

    # Display all the images and their score wrt to the text prompt provided.
    print(f"Searching images by prompt: {args.text}")
    for i in range(len(predictions)):
        print(
            f"\t Image with name: {image_names[i]} has a similarity score={predictions[i]}"
        )

    # Show image
    print("Displaying the most relevant image")

    selected_image = image_names[np.argmax(predictions)]
    if not args.image_dir:
        selected_image = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, selected_image
        )
    else:
        selected_image = os.path.join(args.image_dir, selected_image)
    most_relevant_image = load_image(selected_image)

    if not is_test:
        display_or_save_image(most_relevant_image, args.output_dir)


if __name__ == "__main__":
    main()
