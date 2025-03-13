# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import os
from pathlib import Path

from diffusers import DPMSolverMultistepScheduler

from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized import (
    make_calib_data_unet_vae,
)
from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.model import (
    make_text_encoder_hf_model,
    make_time_embedding_hf_model,
    make_unet_hf_model,
    make_vae_hf_model,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = "stable_diffusion_v1_5_w8a16_quantized"
MODEL_ASSET_VERSION = 3

PROMPT_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "calibration_prompts_500.txt"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--num-samples-unet",
        type=int,
        default=100,
        help="Number of prompts. Final number of sample is multiplied by 2*num-steps.",
    )
    parser.add_argument(
        "--num-samples-vae",
        type=int,
        default=500,
        help="Number of VaeDecoder calibration samples to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where dataset should be stored. Defaults to ./build.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path() / "build")
    os.makedirs(output_dir, exist_ok=True)

    text_encoder_hf = make_text_encoder_hf_model()
    time_embedding_hf = make_time_embedding_hf_model()
    unet_hf = make_unet_hf_model()
    vae_hf = make_vae_hf_model()
    scheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    make_calib_data_unet_vae(
        output_dir,
        text_encoder_hf,
        unet_hf,
        time_embedding_hf,
        scheduler,
        num_steps=args.num_steps,
        num_samples_unet=args.num_samples_unet,
        num_samples_vae=args.num_samples_vae,
    )
