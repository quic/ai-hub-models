# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from datasets import load_dataset

from qai_hub_models.models.huggingface_wavlm_base_plus.app import (
    HuggingFaceWavLMBasePlusApp,
)
from qai_hub_models.models.huggingface_wavlm_base_plus.model import (
    HuggingFaceWavLMBasePlus,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args

HUGGINGFACE_WAVLM_DATASET = "hf-internal-testing/librispeech_asr_demo"


# Run HuggingFace WavLM on a sample audio input, and produce
# a feature vector from the audio. The feature vector will be printed to terminal
def demo_main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(HuggingFaceWavLMBasePlus)
    args = parser.parse_args([] if is_test else None)

    # load model
    model = model_from_cli_args(HuggingFaceWavLMBasePlus, args)

    # Load Application
    app = HuggingFaceWavLMBasePlusApp(model)

    # Load audio
    dataset = load_dataset(HUGGINGFACE_WAVLM_DATASET, "clean", split="validation")
    audio = [x["array"] for x in dataset[:2]["audio"]][0]

    feature_vec = app.predict_features(input=audio)

    # Get output from model
    if not is_test:
        print("Feature vec from audio:\n")
        print(feature_vec)
        print("\n")


if __name__ == "__main__":
    demo_main()
