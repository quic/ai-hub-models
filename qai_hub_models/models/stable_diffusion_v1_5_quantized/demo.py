# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from diffusers import DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer

from qai_hub_models.models._shared.stable_diffusion.demo import stable_diffusion_demo
from qai_hub_models.models.stable_diffusion_v1_5_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    ClipVITTextEncoder,
    Unet,
    VAEDecoder,
)


# Run Stable Diffuison end-to-end on a given prompt. The demo will output an
# AI-generated image based on the description in the prompt.
def main(is_test: bool = False):
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", subfolder="", revision="main"
    )

    scheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    time_embedding = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    ).time_embedding

    text_encoder = ClipVITTextEncoder.from_precompiled()
    unet = Unet.from_precompiled()
    vae_decoder = VAEDecoder.from_precompiled()
    stable_diffusion_demo(
        model_id=MODEL_ID,
        model_asset_version=MODEL_ASSET_VERSION,
        text_encoder=text_encoder,
        unet=unet,
        vae_decoder=vae_decoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        time_embedding=time_embedding,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
