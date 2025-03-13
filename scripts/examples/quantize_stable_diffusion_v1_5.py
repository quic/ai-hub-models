# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to compute AIMET encodings for Stable
Diffusion v1.5 using sample prompts.

Please run `make_calib_data_stable_diffusion_v1_5.py` first to generate the
calibration data to provide as --calib-path.

This script assumes the model is added to QAISM, but is missing quantization parameters.
"""
import argparse
import os
from pathlib import Path

import torch

from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.model import (
    TextEncoderQuantizable,
    UnetQuantizable,
    VaeDecoderQuantizable,
    load_unet_calib_dataset_entries,
    load_vae_calib_dataset_entries,
)
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader

COMPONENTS = {
    "text_encoder": TextEncoderQuantizable,
    "unet": UnetQuantizable,
    "vae_decoder": VaeDecoderQuantizable,
}

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=COMPONENTS.keys(),
        required=True,
        help="One of " + ", ".join(COMPONENTS.keys()),
    )
    parser.add_argument(
        "--calib-path",
        type=str,
        default=None,
        help=(
            "Local path to calib files for the component. "
            "e.g. unet_calib_n{num_steps}_t{num_samples}.npz. "
            "Not needed for text_encoder. "
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where encodings should be stored. Defaults to ./build.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Encodings filename. Defaults to <component>.encodings.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples used to calibrate, Default None to use all available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="For reproducibility.",
    )
    args = parser.parse_args()

    if args.component in ["unet", "vae_decoder"] and args.calib_path is None:
        raise ValueError(f"--calib-path is required for component {args.component}")

    torch.manual_seed(args.seed)

    output_dir = args.output_dir or str(Path() / "build")
    os.makedirs(output_dir, exist_ok=True)
    output_name = args.output_name or args.component

    model_cls = COMPONENTS[args.component]
    model_quant = model_cls.from_pretrained(aimet_encodings=None)

    if args.component == "text_encoder":
        ds = model_quant.get_calibration_data()
    elif args.component == "unet":
        ds = load_unet_calib_dataset_entries(path=args.calib_path)
    elif args.component == "vae_decoder":
        ds = load_vae_calib_dataset_entries(path=args.calib_path)

    data_loader = dataset_entries_to_dataloader(ds)

    model_quant.quantize(data_loader, num_samples=args.num_samples)
    model_quant.convert_to_onnx_and_aimet_encodings(
        output_dir=output_dir, model_name=output_name
    )
