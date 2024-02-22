# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse

import numpy as np
from PIL import Image

from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeHand,
)
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.camera_capture import capture_and_display_processed_frames
from qai_hub_models.utils.display import display_or_save_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hand.jpeg"
)


# Run Mediapipe Hand landmark detection end-to-end on a sample image or camera stream.
# The demo will display output with the predicted landmarks & bounding boxes drawn.
def main(is_test: bool = False):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="image file path or URL",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera Input ID",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.95,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    add_output_dir_arg(parser)

    print(
        "Note: This readme is running through torch, and not meant to be real-time without dedicated ML hardware."
    )
    print("Use Ctrl+C in your terminal to exit.")

    args = parser.parse_args([] if is_test else None)
    if is_test:
        args.image = INPUT_IMAGE_ADDRESS

    # Load app
    app = MediaPipeHandApp(
        MediaPipeHand.from_pretrained(), args.score_threshold, args.iou_threshold
    )
    print("Model and App Loaded")

    if args.image:
        image = load_image(args.image)
        pred_image = app.predict_landmarks_from_image(image)
        out_image = Image.fromarray(pred_image[0], "RGB")
        if not is_test:
            display_or_save_image(out_image, args.output_dir)
    else:

        def frame_processor(frame: np.ndarray) -> np.ndarray:
            return app.predict_landmarks_from_image(frame)[0]  # type: ignore

        capture_and_display_processed_frames(
            frame_processor, "QAIHM Mediapipe Hand Demo", args.camera
        )


if __name__ == "__main__":
    main()
