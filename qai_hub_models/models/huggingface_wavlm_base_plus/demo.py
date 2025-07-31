# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.huggingface_wavlm_base_plus.app import (
    HuggingFaceWavLMBasePlusApp,
)
from qai_hub_models.models.huggingface_wavlm_base_plus.model import (
    MODEL_ID,
    SAMPLE_INPUTS,
    HuggingFaceWavLMBasePlus,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_numpy


# Run HuggingFace WavLM on a sample audio input, and produce
# a feature vector from the audio. The feature vector will be printed to terminal
def demo_main(is_test: bool = False):
    parser = get_model_cli_parser(HuggingFaceWavLMBasePlus)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--audio",
        type=str,
        default=SAMPLE_INPUTS,
        help="audio file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(HuggingFaceWavLMBasePlus, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load Application
    app = HuggingFaceWavLMBasePlusApp(model)

    # Load audio
    audio = load_numpy(args.audio)["audio"]

    feature_vec = app.predict_features(input=audio)

    # Get output from model
    if not is_test:
        print("Transcriptions from audio:\n")
        print(feature_vec)
        print("\n")


if __name__ == "__main__":
    demo_main()
