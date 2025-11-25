# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from mobile_sam.utils.transforms import ResizeLongestSide

from qai_hub_models.models._shared.sam.app import SAMApp, SAMInputImageLayout
from qai_hub_models.models.mobilesam.model import (
    DEFAULT_MODEL_TYPE,
    MobileSAM,
)
from qai_hub_models.models.track_anything.app import TrackAnythingApp
from qai_hub_models.models.track_anything.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    TrackAnythingWrapper,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

VIDEO_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "demo.mp4"
)


def generate_video_from_frames(
    frames: list[np.ndarray], output_path: str, fps: int = 30
) -> None:
    """
    Generates a video from a list of frames.

    Parameters
    ----------
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    height, width, _layers = frames[0].shape
    fourcc = 0x39307076  # hex code for "vp09" format
    output_path = os.path.join(Path.cwd(), "build", output_path)
    print("Saving image to ", output_path)
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()


def generate_frames_from_video(video_path: str) -> list[np.ndarray]:
    """
    Generates list of frames from a video.

    Parameters
    ----------
        video_path (str): The path of the input video.

    Returns
    -------
        frames (list of numpy array): frames generated from a video.
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_memory_usage = psutil.virtual_memory().percent
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if current_memory_usage > 90:
                print(
                    "Memory usage is too high (>90%). Please reduce the video resolution or frame rate."
                )
                break
        else:
            break
    return frames


def main(is_test: bool = False):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        help="video file path or URL.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        help=f"SAM model type to load. Tested with model type `{DEFAULT_MODEL_TYPE}`.",
    )
    parser.add_argument(
        "--point-coordinates",
        type=str,
        default="280,120;",
        help="Comma separated x and y coordinate. Multiple coordinate separated by `;`."
        " e.g. `x1,y1;x2,y2`. Default: `280,120;`",
    )
    parser.add_argument(
        "--single-mask-mode",
        type=bool,
        default=True,
        help="If True, returns single mask. For multiple points multiple masks could lead to better results.",
    )
    args = parser.parse_args(["--model-type", DEFAULT_MODEL_TYPE] if is_test else None)

    coordinates: list[str] = list(filter(None, args.point_coordinates.split(";")))

    # Load SAM Application
    sam_wrapper = MobileSAM.from_pretrained(args.model_type)
    sam_app = SAMApp(
        sam_wrapper.sam.image_encoder.img_size,
        sam_wrapper.sam.mask_threshold,
        SAMInputImageLayout[sam_wrapper.sam.image_format],
        [sam_wrapper.encoder],
        sam_wrapper.decoder,
        ResizeLongestSide,
    )

    video_input = str(VIDEO_ADDRESS.fetch()) if args.video is None else str(args.video)

    # Get first frame
    frames = generate_frames_from_video(video_input)
    first_frame = frames[0]

    # Point segmentation using sam decoder
    print("\n** Performing point segmentation **\n")

    # Input points
    input_coords = []
    input_labels = []

    for coord in coordinates:
        coord_split = coord.split(",")
        if len(coord_split) != 2:
            raise RuntimeError(
                f"Expecting comma separated x and y coordinate. Provided {coord_split}."
            )

        input_coords.append([int(coord_split[0]), int(coord_split[1])])
        # Set label to `1` to include current point for segmentation
        input_labels.append(1)

    # Generate masks with given input points
    generated_mask_pt, *_ = sam_app.predict_mask_from_points(
        first_frame, torch.Tensor(input_coords), torch.Tensor(input_labels)
    )
    generated_mask = generated_mask_pt.squeeze(0).squeeze(0).numpy()

    # load TrackAnything Application
    wrapper = TrackAnythingWrapper.from_pretrained()
    app = TrackAnythingApp(
        wrapper.EncodeKeyWithShrinkage,
        wrapper.EncodeValue,
        wrapper.EncodeKeyWithoutShrinkage,
        wrapper.Segment,
        wrapper.config,
    )

    # Run inference
    print("\n** Tracking object... **\n")
    painted_images = app.track(frames, generated_mask)

    if not is_test:
        generate_video_from_frames(
            painted_images, output_path="out_painted_image.mp4", fps=30
        )


if __name__ == "__main__":
    main()
