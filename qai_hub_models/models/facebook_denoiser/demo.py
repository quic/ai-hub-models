# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import tempfile
from typing import List

from qai_hub_models.models.facebook_denoiser.app import FacebookDenoiserApp
from qai_hub_models.models.facebook_denoiser.model import (
    ASSET_VERSION,
    MODEL_ID,
    SAMPLE_RATE,
    FacebookDenoiser,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_path

EXAMPLE_RECORDING = CachedWebModelAsset.from_asset_store(
    MODEL_ID, ASSET_VERSION, "icsi_meeting_recording.wav"
)


def main(is_test: bool = False):
    """
    Run facebook denoiser on a sample audio (`.wav`) file.
    """
    parser = get_model_cli_parser(FacebookDenoiser)
    parser.add_argument(
        "--audio",
        nargs="+",
        default=[EXAMPLE_RECORDING],
        help="WAV file paths or URLs",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help="Audio sample rate the model was trained on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getcwd(),
        help="output directory (where output WAV should be written)",
    )
    args = parser.parse_args([] if is_test else None)

    # Load Model
    source_model = model_from_cli_args(FacebookDenoiser, args)
    app = FacebookDenoiserApp(source_model, args.sample_rate)

    # Download data
    audio: List[str] = args.audio
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, file in enumerate(audio):
            audio[idx] = load_path(file, tmpdir)

        # Dump output from app
        output = app.denoise_audio(audio, args.output_dir)

    if not is_test:
        print("Wrote files:")
        for path in output:
            print(str(path))


if __name__ == "__main__":
    main()
