# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import importlib
import os
from pathlib import Path

from qai_hub_models.models._shared.stable_diffusion.utils import (
    make_calib_data_unet_vae,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_IDS = [
    "stable_diffusion_v1_5_w8a16_quantized",
    "stable_diffusion_v2_1_quantized",
    "stable_diffusion_turbo",
]

PROMPT_PATH = CachedWebModelAsset.from_asset_store(
    "stable_diffusion_v1_5_w8a16_quantized", 4, "calibration_prompts_500.txt"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        choices=MODEL_IDS,
        required=True,
        help="One of " + ", ".join(MODEL_IDS),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of diffusion steps. Default is model-dependent.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help=(
            "Number of prompts to use. It generates "
            "num_samples * num_steps * 2 samples for unet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where dataset should be stored. Defaults to ./build.",
    )
    args = parser.parse_args()
    model_id = args.model_id

    # Dynamically import the module
    model_cls = importlib.import_module(f"qai_hub_models.models.{model_id}").Model

    num_steps = args.num_steps or model_cls.default_num_steps
    print(f"{num_steps=}")

    TextEncoderQuantizable = model_cls.component_classes[0]
    UnetQuantizable = model_cls.component_classes[1]
    VaeDecoderQuantizable = model_cls.component_classes[2]

    tokenizer = model_cls.make_tokenizer()
    scheduler = model_cls.make_scheduler()

    output_dir = args.output_dir or str(Path() / "build" / model_id)
    os.makedirs(output_dir, exist_ok=True)

    text_encoder_hf = TextEncoderQuantizable.make_adapted_torch_model()
    unet_hf = UnetQuantizable.make_adapted_torch_model()
    vae_hf = VaeDecoderQuantizable.make_adapted_torch_model()

    make_calib_data_unet_vae(
        output_dir,
        PROMPT_PATH.fetch(),
        tokenizer,
        text_encoder_hf,
        unet_hf,
        scheduler,
        num_steps=num_steps,
        num_samples=args.num_samples,
        guidance_scale=model_cls.guidance_scale,
    )
