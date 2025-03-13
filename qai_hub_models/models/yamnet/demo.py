# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.yamnet.app import YamNetApp
from qai_hub_models.models.yamnet.model import INPUT_AUDIO_ADDRESS, MODEL_ID, YamNet
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_path, qaihm_temp_dir


#
# Run YamNetApp end-to-end on a sample audio.
# The demo will display top 5 classification predictions for the audio.
#
def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(YamNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)
    parser.add_argument(
        "--audio", type=str, default=INPUT_AUDIO_ADDRESS, help="audio file path or URL."
    )

    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(YamNet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)
    # Load audio & model
    app = YamNetApp(model)
    print("Model Loaded")
    with qaihm_temp_dir() as tmpdir:
        dst_path = load_path(args.audio, tmpdir)
        predictions = app.predict(path=str(dst_path))
    top5_classes = " | ".join(predictions)
    if not is_test:
        print(f"Top 5 predictions: {top5_classes}")


if __name__ == "__main__":
    main()
