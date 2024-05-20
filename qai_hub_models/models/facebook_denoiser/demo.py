# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from pathlib import Path
from typing import List

import torchaudio

from qai_hub_models.models.facebook_denoiser.app import FacebookDenoiserApp
from qai_hub_models.models.facebook_denoiser.model import (
    ASSET_VERSION,
    DEFAULT_SEQUENCE_LENGTH,
    MODEL_ID,
    SAMPLE_RATE,
    FacebookDenoiser,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_path,
    qaihm_temp_dir,
)

EXAMPLE_RECORDING = CachedWebModelAsset.from_asset_store(
    MODEL_ID, ASSET_VERSION, "icsi_meeting_recording.wav"
)


def main(is_test: bool = False):
    """
    Run facebook denoiser on a sample audio (`.wav`) file.
    """
    parser = get_model_cli_parser(FacebookDenoiser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
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
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(FacebookDenoiser, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    app = FacebookDenoiserApp(model, args.sample_rate)

    # Download data
    audio_files: List[str] = args.audio
    audio_tensors = []
    with qaihm_temp_dir() as tmpdir:
        for idx, file in enumerate(audio_files):
            audio_file = load_path(file, tmpdir)
            audio, sample_rate = torchaudio.load(audio_file)
            # By default, cut audio to the default sequence length
            # since by default, model is compiled with this input size
            audio_tensor = audio[0, :DEFAULT_SEQUENCE_LENGTH].unsqueeze(0).unsqueeze(0)
            assert sample_rate == SAMPLE_RATE
            audio_tensors.append(audio_tensor)

        # Dump output from app
        output = app.denoise_audio(audio_tensors)

        if args.output_dir:
            output_files = []
            for file, estimate in zip(audio_files, output):
                local_path = load_path(file, tmpdir)
                filename = os.path.join(
                    args.output_dir, os.path.basename(local_path).rsplit(".", 1)[0]
                )
                filename = Path(f"{filename}_enhanced.wav")
                # make input 2D:
                estimate = estimate.squeeze().unsqueeze(0)
                torchaudio.save(filename, estimate, SAMPLE_RATE)
                output_files.append(filename)
            return output_files

    if not is_test:
        print("Wrote files:")
        for path in output:
            print(str(path))


if __name__ == "__main__":
    main()
