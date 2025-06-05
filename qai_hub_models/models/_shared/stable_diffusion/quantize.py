# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from pathlib import Path

import torch

from qai_hub_models.models._shared.stable_diffusion.model import StableDiffusionBase
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.quantization import get_calibration_data


def stable_diffusion_quantize(
    model_cls: type[StableDiffusionBase], model_id: str, default_num_steps: int
):
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=model_cls.component_class_names,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="DEFAULT_UNQUANTIZED",
        help="Huggingface repo id or local directory with custom weights.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=f"Directory where quantized checkpoint should be stored. Defaults to ./build/{model_id}/<component>.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples used to calibrate, Default None to use all available.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=default_num_steps,
        help="Number of samples used to calibrate, Default None to use all available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="For reproducibility.",
    )
    parser.add_argument(
        "--server-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=("One of cpu,cuda. Run QuantSim calibration on this server device. "),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    host_device = torch.device(args.host_device)
    component_cls = dict(
        zip(model_cls.component_class_names, model_cls.component_classes)
    )[args.component]
    component = component_cls.from_pretrained(
        checkpoint=args.checkpoint, host_device=host_device
    )

    dataset_options = dict(
        sd_cls=model_cls,
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
    )

    # get_calibration_data is also used in submit_quantize_job for non-aimet
    # models
    ds = get_calibration_data(
        component, num_samples=args.num_samples, dataset_options=dataset_options
    )
    data_loader = dataset_entries_to_dataloader(ds)

    component.quantize(data_loader, num_samples=args.num_samples)

    output_dir = args.output or str(Path() / "build" / model_id)
    component.save_calibrated_checkpoint(output_checkpoint=output_dir)
