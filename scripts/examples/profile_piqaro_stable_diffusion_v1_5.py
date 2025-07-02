# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to profile various optimizations for Stable
Diffusion v1.5.

Install piqaro from https://github.qualcomm.com/Hexagon-Architecture/piqaro

This is broken currently. Will fix soon.
"""
import argparse
import logging
import os
import tempfile
from pathlib import Path

import piqaro
import piqaro.onnx
import qai_hub as hub
import torch

from qai_hub_models.models.stable_diffusion_v1_5.model import (
    TextEncoderQuantizable,
    UnetQuantizable,
    VaeDecoderQuantizable,
)
from qai_hub_models.utils.base_model import Precision, TargetRuntime
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.onnx_helpers import (
    torch_onnx_export_with_large_model_size_check,
)
from qai_hub_models.utils.printing import print_profile_metrics_from_job
from qai_hub_models.utils.qai_hub_helpers import export_torch_to_onnx_zip

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

COMPONENTS = {
    "text_encoder": TextEncoderQuantizable,
    "unet": UnetQuantizable,
    "vae_decoder": VaeDecoderQuantizable,
}

OPT_METHODS = ["no_opt", "manual", "piqaro_torch", "piqaro_onnx"]


def piqaro_onnx_large_model(onnx_model, sample_input, export_dir):
    import onnx
    import onnxsim

    onnx_model, _ = onnxsim.simplify(onnx_model)
    # Convert to piQaro/PyTorch format
    torch_model = piqaro.onnx._acquire(onnx_model)

    # Optimize
    opt = piqaro.Optimizer()
    opt(torch_model)

    # Export back to ONNX
    # For models > 2GB, must specify an absolute path so weight files
    # can be written next to the .onnx file
    onnx_path = os.path.join(export_dir, "model.onnx")
    logger.info(f"Saving piqaro-onnx optimized ONNX model to {onnx_path}")

    torch_onnx_export_with_large_model_size_check(torch_model, sample_input, onnx_path)

    onnx_model = onnx.load(onnx_path)
    return onnx_model


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=COMPONENTS.keys(),
        default="unet",
        help="One of " + ", ".join(COMPONENTS.keys()),
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="no_opt",
        help="Optimization method. One of {OPT_METHODS}. Default is no_opt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where ONNX files are stored. Defaults to "
            "./build/{component}_{opt}.onnx.zip"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="For reproducibility.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_cls = COMPONENTS[args.component]
    component = model_cls.from_pretrained()
    input_spec = component.get_input_spec()

    assert args.opt in OPT_METHODS, f"Unsupported {args.opt}"
    apply_monkey_patch = args.opt == "manual"

    if args.component == "unet":
        torch_model = UnetQuantizable.make_adapted_torch_model(
            on_device_opt=apply_monkey_patch
        )
    elif args.component == "text_encoder":
        torch_model = TextEncoderQuantizable.make_adapted_torch_model()
    elif args.component == "vae_decoder":
        torch_model = VaeDecoderQuantizable.make_adapted_torch_model()

    dummy_input = tuple(make_torch_inputs(input_spec))

    # piqaro.config.load_file('toyota.yaml')
    if args.opt == "piqaro_torch":
        # optimized_model = piqaro.onnx.optimize(model)
        logger.info("Optimizing with Piqaro torch")
        torch_model = piqaro.optimize(torch_model, dummy_input)

    # Export to ONNX
    logger.info("Exporting to onnx...")
    output_dir = args.output_dir or str(
        Path() / "build" / f"{args.component}_{args.opt}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path_onnx = os.path.join(
        output_dir, f"sd1_5_{args.component}_{args.opt}.onnx"
    )
    onnx_transforms = None

    if args.opt == "piqaro_onnx":
        # Create a temporary directory that will persist until export finishes.
        with tempfile.TemporaryDirectory() as tmpdir:

            def onnx_transforms(onnx_model):
                # import onnxsim
                # onnx_model, _ = onnxsim.simplify(onnx_model)

                # TODO: simplify after piqaro fixes
                # https://github.qualcomm.com/Hexagon-Architecture/piqaro/issues/914
                # return piqaro.onnx.optimize(onnx_model)
                # export_dir = "/mnt/vol1/tetra/m6/scripts/examples/build/llama_v3_2_3b_piqaro_onnx/piqaro_opt_torch_export.onnx"
                # os.makedirs(export_dir, exist_ok=True)
                return piqaro_onnx_large_model(onnx_model, dummy_input, tmpdir)

            output_dir = export_torch_to_onnx_zip(
                torch_model,
                str(output_dir),
                dummy_input,
                input_names=list(input_spec.keys()),
                onnx_transforms=onnx_transforms,
                skip_zip=True,
            )
    else:
        output_dir = export_torch_to_onnx_zip(
            torch_model,
            str(output_dir),
            dummy_input,
            input_names=list(input_spec.keys()),
            onnx_transforms=onnx_transforms,
            skip_zip=True,
        )

    compile_options = component.get_hub_compile_options(
        TargetRuntime.QNN_CONTEXT_BINARY, precision=Precision.w8a16
    )

    compile_jobs = []
    # Profile on all 3 devices
    devices = [
        "Snapdragon X Elite CRD",
        "Samsung Galaxy S23 (Family)",
        "Samsung Galaxy S24 (Family)",
    ]
    hub_model = hub.upload_model(output_dir)
    profile_jobs = []
    for device in devices:
        hub_device = hub.Device(device)
        compile_job = hub.submit_compile_job(
            model=hub_model,
            input_specs=input_spec,
            device=hub_device,
            name=f"sd1_5_{args.component}_{args.opt}",
            options=compile_options,
        )
        logger.info(f"compile job: {compile_job}")

        profile_job = hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=hub_device,
            name=f"sd1_5_{args.component}_{args.opt}",
        )
        logger.info(f"profile job: {profile_job}")
        profile_jobs.append(profile_job)

    for profile_job in profile_jobs:
        assert profile_job.wait().success, "Job failed: " + profile_job.url
        profile_data = profile_job.download_profile()
        logger.info(print_profile_metrics_from_job(profile_job, profile_data))
