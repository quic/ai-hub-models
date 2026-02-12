# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage

from qai_hub_models.models.fomm.app import FOMMApp, load_video_frames, resize_frame
from qai_hub_models.models.fomm.model import FOMM, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_path,
    qaihm_temp_dir,
)
from qai_hub_models.utils.evaluate import EvalMode

SOURCE_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_source_image.png"
)
DRIVING_VIDEO_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_driving_video.mp4"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(FOMM)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

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
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of frames to process",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load inputs
    source_image = np.array(load_image(args.source_image)) / 255.0

    with qaihm_temp_dir() as tmpdir:
        video_path = load_path(args.driving_video, tmpdir)
        all_frames = load_video_frames(video_path)
        driving_frames = (
            all_frames if args.num_frames < 0 else all_frames[: args.num_frames]
        )

    print(f"Loaded {len(driving_frames)} driving frames.")

    python_model = model_from_cli_args(FOMM, args)
    app = FOMMApp(python_model)

    if args.eval_mode == EvalMode.ON_DEVICE:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required for on-device mode.")

        detector_compiled, generator_compiled = demo_model_components_from_cli_args(
            FOMM, MODEL_ID, args
        )

        app.kp_detector = detector_compiled  # type: ignore[assignment]
        app.generator = generator_compiled  # type: ignore[assignment]

    print("Running FOMM animation...")
    output_frames = app.copy_motion_to_video(source_image, driving_frames)

    # Visualization
    resized_source = resize_frame(source_image)
    side_by_side = [
        np.concatenate([src, resized_source, out], axis=1)
        for src, out in zip(driving_frames, output_frames, strict=False)
    ]

    if is_test:
        return

    build_dir = Path.cwd() / "build" / MODEL_ID
    build_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir or build_dir) / "fomm_output.gif"
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis("off")
    im = ax.imshow(side_by_side[0])

    def update(frame: np.ndarray) -> list[AxesImage]:
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=side_by_side, interval=50, blit=True)
    ani.save(str(out_path), writer="pillow", fps=20)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
