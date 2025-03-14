# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
from diffusers import DPMSolverMultistepScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTokenizer

from qai_hub_models.models._shared.stable_diffusion.app import StableDiffusionApp
from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.export import (
    ALL_COMPONENTS,
)
from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.model import (
    MODEL_ID,
    TextEncoderQuantizable,
    UnetQuantizable,
    VaeDecoderQuantizable,
    make_text_encoder_hf_model,
    make_unet_hf_model,
    make_vae_hf_model,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.display import display_or_save_image

DEFAULT_PROMPT = "Painting - She Danced By The Light Of The Moon by Steve Henderson"


# Run Stable Diffuison end-to-end on a given prompt. The demo will output an
# AI-generated image based on the description in the prompt.
def main(is_test: bool = False):
    parser = get_on_device_demo_parser(
        available_target_runtimes=[TargetRuntime.QNN],
        default_device="Snapdragon X Elite CRD",
        add_output_dir=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt for stable diffusion",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--use-torch-fp32", action="store_true", help="Use torch fp32 (no AIMET)"
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    if args.use_torch_fp32:
        text_encoder = make_text_encoder_hf_model()
        unet = make_unet_hf_model()
        vae_decoder = make_vae_hf_model()
    else:
        model_cls = [
            TextEncoderQuantizable,
            UnetQuantizable,
            VaeDecoderQuantizable,
        ]
        text_encoder, unet, vae_decoder = demo_model_components_from_cli_args(
            model_cls, MODEL_ID, ALL_COMPONENTS, args
        )

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

    app = StableDiffusionApp(
        text_encoder=text_encoder,
        vae_decoder=vae_decoder,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        time_embedding=time_embedding,
        channel_last_latent=False,  # OnDeviceModel account for channel_last already
    )

    image = app.generate_image(
        args.prompt,
        num_steps=args.num_steps,
    )

    pil_img = Image.fromarray(
        np.round(image.detach().cpu().numpy() * 255).astype(np.uint8)[0]
    )

    if args.use_torch_fp32:
        default_output_dir = "export/torch_fp32"
    elif args.on_device:
        default_output_dir = "export/on_device_e2e"
    else:
        default_output_dir = "export/quantsim"
    output_dir = args.output_dir or default_output_dir
    display_or_save_image(pil_img, output_dir)


if __name__ == "__main__":
    main()
