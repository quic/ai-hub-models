# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from qai_hub_models.models.fomm.app import FOMMApp, load_video_frames, resize_frame
from qai_hub_models.models.fomm.model import FOMM, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_path,
    qaihm_temp_dir,
)

SOURCE_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_source_image.png"
)
DRIVING_VIDEO_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_driving_video.mp4"
)


def main(is_test: bool = False):
    # Read inputs
    parser = get_model_cli_parser(FOMM)
    parser.add_argument(
        "--source-image",
        type=str,
        default=SOURCE_IMAGE_ADDRESS,
        help="Filepath or URL to the image to animate.",
    )
    parser.add_argument(
        "--driving-video",
        type=str,
        default=DRIVING_VIDEO_ADDRESS,
        help="Filepath or URL to video of intended animation.",
    )
    args = parser.parse_args([] if is_test else None)
    source_image = load_image(args.source_image)
    with qaihm_temp_dir() as tmpdir:
        driving_video_path = load_path(args.driving_video, tmpdir)
        driving_video_frames = load_video_frames(
            driving_video_path, sample_rate=10 if is_test else 1
        )

    # run the model
    app = FOMMApp(model_from_cli_args(FOMM, args))
    source_image_arr = np.array(source_image)
    output_image_list = app.copy_motion_to_video(source_image_arr, driving_video_frames)
    resized_source = resize_frame(np.array(source_image_arr))
    concat_frames = [
        np.concatenate((frame1, resized_source, frame2), axis=1)
        for frame1, frame2 in zip(driving_video_frames, output_image_list)
    ]

    if not is_test:
        fig, ax = plt.subplots()
        ax.axis(False)
        im = plt.imshow(concat_frames[0], cmap=plt.get_cmap("jet"), vmin=0, vmax=255)

        def updatefig(j):
            im.set_array(concat_frames[j])
            return [im]

        ani = FuncAnimation(  # noqa: F841
            fig, updatefig, frames=len(concat_frames), interval=50, blit=True
        )
        plt.show()


if __name__ == "__main__":
    main()
