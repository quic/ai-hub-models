# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from huggingface_hub import hf_hub_download

from qai_hub_models.models._shared.stable_diffusion.model import StableDiffusionBase
from qai_hub_models.utils.checkpoint import CheckpointSpec, hf_repo_exists
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.quantization import get_calibration_data


def maybe_save_scheduler_config(checkpoint: CheckpointSpec, output_dir: str | Path):
    """
    Save the scheduler config from a HuggingFace repo to the output directory.

    Args:
        checkpoint: Hugging Face repo ID or local path.
        output_dir: Directory where the scheduler config should be saved.
    """

    scheduler_dir = Path(output_dir) / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    target_path = scheduler_dir / SCHEDULER_CONFIG_NAME
    if target_path.exists():
        return  # Already exists
    if not hf_repo_exists(str(checkpoint)):
        return
    config_path = hf_hub_download(
        repo_id=str(checkpoint),
        filename=f"scheduler/{SCHEDULER_CONFIG_NAME}",
    )
    shutil.copy(config_path, target_path)
    print(f"Scheduler config saved to {target_path}")


def stable_diffusion_quantize(
    model_cls: type[StableDiffusionBase],
    model_id: str,
    default_num_steps: int,
    use_controlnet: bool = False,
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
        default=100,
        help="Number of samples used to calibrate.",
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
        "--host-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=("One of cpu,cuda. Run QuantSim calibration on this host device. "),
    )

    # --prompt and --image-cond are required for controlnet
    if use_controlnet:
        kwargs: dict[str, Any] = {"required": True}
    else:
        kwargs = {"default": ""}
    parser.add_argument(
        "--prompt",
        type=str,
        help=(
            "Path to a plain text file where each line is a prompt. "
            "The default uses a preset of 500 prompts"
        ),
        **kwargs,
    )
    if use_controlnet:
        parser.add_argument(
            "--image-cond",
            type=str,
            required=True,
            help=(".pt path containing a list of canny torch.Tensor NCHW images"),
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
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        checkpoint=args.checkpoint,
        use_controlnet=use_controlnet,
        prompt_path=args.prompt,
        image_cond_path=args.image_cond,
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

    checkpoint = model_cls.handle_default_checkpoint(args.checkpoint)
    maybe_save_scheduler_config(checkpoint, output_dir)
