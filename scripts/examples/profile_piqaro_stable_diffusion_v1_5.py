# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to profile various optimizations for Stable
Diffusion v1.5.

Install piqaro from https://github.qualcomm.com/Hexagon-Architecture/piqaro
"""
import argparse
import os
from pathlib import Path

import piqaro
import piqaro.onnx
import qai_hub as hub
import torch

from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.model import (
    TextEncoderQuantizable,
    UnetQuantizable,
    VaeDecoderQuantizable,
    make_text_encoder_hf_model,
    make_unet_hf_model,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.printing import print_profile_metrics_from_job
from qai_hub_models.utils.qai_hub_helpers import export_torch_to_onnx_zip

COMPONENTS = {
    "text_encoder": TextEncoderQuantizable,
    "unet": UnetQuantizable,
    "vae_decoder": VaeDecoderQuantizable,
}

OPT_METHODS = ["no_opt", "manual", "piqaro_torch", "piqaro_onnx"]

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
        "--opt",
        type=str,
        default="no_opt",
        help="Optimization method. One of {OPT_METHODS}. Default is no_opt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Snapdragon X Elite CRD",
        help="Hub device",
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

    hub_device = hub.Device(args.device)

    if args.component == "unet":
        torch_model = make_unet_hf_model(apply_monkey_patch=apply_monkey_patch)
    elif args.component == "text_encoder":
        torch_model = make_text_encoder_hf_model()

    dummy_input = tuple(make_torch_inputs(input_spec))
    if args.opt == "piqaro_torch":
        print("Optimizing with Piqaro torch")
        torch_model = piqaro.optimize(torch_model, dummy_input)

    # Export to ONNX
    print("Exporting to onnx...")
    output_dir = args.output_dir or str(
        Path() / "build" / f"{args.component}_{args.opt}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path_onnx = os.path.join(
        output_dir, f"sd1_5_{args.component}_{args.opt}.onnx"
    )
    onnx_transforms = None
    if args.opt == "piqaro_onnx":

        def onnx_transforms(onnx_model):
            import onnxsim

            onnx_model, _ = onnxsim.simplify(onnx_model)
            return piqaro.onnx.optimize(onnx_model)

    zip_path = export_torch_to_onnx_zip(
        torch_model,
        output_path_onnx,
        dummy_input,
        input_names=list(input_spec.keys()),
        onnx_transforms=onnx_transforms,
    )

    compile_options = component.get_hub_compile_options(TargetRuntime.QNN)

    compile_job = hub.submit_compile_job(
        model=zip_path,
        input_specs=input_spec,
        device=hub_device,
        name=f"sd1_5_{args.component}_{args.opt}",
        options=compile_options,
    )
    print(f"compile job: {compile_job}")

    profile_job = hub.submit_profile_job(
        model=compile_job.get_target_model(),
        device=hub_device,
        name=f"sd1_5_{args.component}_{args.opt}",
    )
    print(f"profile job: {profile_job}")

    assert profile_job.wait().success, "Job failed: " + profile_job.url
    profile_data = profile_job.download_profile()
    print_profile_metrics_from_job(profile_job, profile_data)
