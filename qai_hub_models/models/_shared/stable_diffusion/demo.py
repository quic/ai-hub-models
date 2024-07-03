# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from typing import Any

import diffusers
import numpy as np
import qai_hub as hub
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
from transformers import CLIPTokenizer

from qai_hub_models.models._shared.stable_diffusion.app import StableDiffusionApp
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.base_model import BasePrecompiledModel
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.inference import OnDeviceModel, get_uploaded_precompiled_model
from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub

DEFAULT_DEMO_PROMPT = "spectacular view of northern lights from Alaska"
DEFAULT_DEVICE_NAME = "Samsung Galaxy S23 Ultra"


def _get_on_device_model(
    model_id: str,
    model_asset_version: str,
    input_model: BasePrecompiledModel,
    model_name: str,
    ignore_cached_model: bool = False,
    device_name=DEFAULT_DEVICE_NAME,
) -> OnDeviceModel:
    if not can_access_qualcomm_ai_hub():
        raise RuntimeError(
            "Stable-diffusion on-device demo requires access to QAI-Hub.\n"
            "Please visit https://aihub.qualcomm.com/ and sign-up."
        )
    # Upload model
    uploaded_model = get_uploaded_precompiled_model(
        input_model.get_target_model_path(),
        model_id,
        model_asset_version,
        model_name,
        ignore_cached_model=ignore_cached_model,
    )
    inputs = list(input_model.get_input_spec().keys())
    return OnDeviceModel(uploaded_model, inputs, hub.Device(name=device_name))


# Run Stable Diffuison end-to-end on a given prompt. The demo will output an
# AI-generated image based on the description in the prompt.
def stable_diffusion_demo(
    model_id: str,
    model_asset_version: str,
    text_encoder: BasePrecompiledModel,
    unet: BasePrecompiledModel,
    vae_decoder: BasePrecompiledModel,
    tokenizer: CLIPTokenizer | Any,
    scheduler: DPMSolverMultistepScheduler,
    time_embedding: diffusers.embeddings.TimeEmbedding,
    channel_last_latent: bool = True,
    is_test: bool = False,
):
    """
    Generate an image by running text_encoder, unet, vae_decoder via AI Hub
    inference job on target physical device, and tokenizer, scheduler, and
    time embedding in torch locally.

    See parser arguments for parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default=DEFAULT_DEMO_PROMPT,
        help="Prompt to generate image from.",
    )
    parser.add_argument(
        "--num-steps",
        default=5,
        type=int,
        help="The number of diffusion steps (higher means better quality).",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed.",
    )
    add_output_dir_arg(parser)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Strength of guidance (higher means more influence from prompt).",
    )
    parser.add_argument(
        "--ignore-cached-model",
        action="store_true",
        help="Uploads model ignoring previously uploaded and cached model.",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        default=DEFAULT_DEVICE_NAME,
        help="Device to run stable-diffusion demo on.",
    )
    args = parser.parse_args([] if is_test else None)

    if not is_test:
        print(f"\n{'-' * 100}")
        print(
            f"** Performing image generation on-device({args.device_name}) with Stable Diffusion **"
        )
        print()
        print("Prompt:", args.prompt)
        print("Number of steps:", args.num_steps)
        print("Guidance scale:", args.guidance_scale)
        print("Seed:", args.seed)
        print()
        print(
            "Note: This reference demo uses significant amounts of memory and may take 4-5 minutes to run ** per step **."
        )
        print(f"{'-' * 100}\n")

    print(f"Downloading model assets\n{'-' * 35}")
    # Load target models

    # Create three OnDeviceModel instances to prepare for on-device inference.
    # This is similar to initializing PyTorch model to call forward method later.
    # Instead of forward, we later submit inference_jobs on QAI-Hub for
    # on-device evaluation.
    print(f"Uploading model assets on QAI-Hub\n{'-' * 35}")
    hub_text_encoder = _get_on_device_model(
        model_id,
        model_asset_version,
        text_encoder,
        "text_encoder",
        args.ignore_cached_model,
        args.device_name,
    )
    hub_unet = _get_on_device_model(
        model_id,
        model_asset_version,
        unet,
        "unet",
        args.ignore_cached_model,
        args.device_name,
    )
    hub_vae_decoder = _get_on_device_model(
        model_id,
        model_asset_version,
        vae_decoder,
        "vae_decoder",
        args.ignore_cached_model,
        args.device_name,
    )

    app = StableDiffusionApp(
        text_encoder=hub_text_encoder,
        vae_decoder=hub_vae_decoder,
        unet=hub_unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        time_embedding=time_embedding,
        channel_last_latent=channel_last_latent,
    )

    image = app.generate_image(
        args.prompt,
        num_steps=args.num_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
    )

    pil_img = Image.fromarray(np.round(image.numpy() * 255).astype(np.uint8)[0])

    if not is_test:
        display_or_save_image(pil_img, args.output_dir)
