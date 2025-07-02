# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from PIL import Image

from qai_hub_models.models._shared.stable_diffusion.app import StableDiffusionApp
from qai_hub_models.models._shared.stable_diffusion.model import StableDiffusionBase
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_model_kwargs,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.display import display_or_save_image, to_uint8
from qai_hub_models.utils.evaluate import EvalMode

DEFAULT_PROMPT = "Painting - She Danced By The Light Of The Moon by Steve Henderson"


# Run Stable Diffuison end-to-end on a given prompt. The demo will output an
# AI-generated image based on the description in the prompt.
def stable_diffusion_demo(
    model_id: str,
    model_cls: type[StableDiffusionBase],
    is_test: bool = False,
    default_guidance_scale: float = 7.5,
    default_num_steps: int = 5,
):
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser=parser,
        supported_eval_modes=[EvalMode.QUANTSIM, EvalMode.FP, EvalMode.ON_DEVICE],
        supported_precisions={Precision.w8a16},
        available_target_runtimes=[TargetRuntime.QNN_CONTEXT_BINARY],
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
        default=default_num_steps,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=default_guidance_scale,
        help="Guidance scale",
    )
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, model_id)

    if args.eval_mode == EvalMode.FP:
        model_kwargs = get_model_kwargs(model_cls, vars(args))
        # model = model_cls.from_pretrained(**kwargs)

        text_encoder_cls = model_cls.component_classes[0]
        unet_cls = model_cls.component_classes[1]
        vae_cls = model_cls.component_classes[2]
        text_encoder = text_encoder_cls.torch_from_pretrained(**model_kwargs)
        unet = unet_cls.torch_from_pretrained(**model_kwargs)
        vae_decoder = vae_cls.torch_from_pretrained(**model_kwargs)
    else:
        text_encoder, unet, vae_decoder = demo_model_components_from_cli_args(
            model_cls, model_id, args
        )

    assert issubclass(model_cls, StableDiffusionBase)
    tokenizer = model_cls.make_tokenizer()
    scheduler = model_cls.make_scheduler(args.checkpoint)

    app = StableDiffusionApp(
        text_encoder=text_encoder,
        vae_decoder=vae_decoder,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        # OnDeviceModel account for channel_last already
        channel_last_latent=False,
    )

    image = app.generate_image(
        args.prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )

    pil_img = Image.fromarray(to_uint8(image.detach().cpu().numpy())[0])

    if args.eval_mode == EvalMode.FP:
        default_output_dir = "export/torch_fp32"
    elif args.eval_mode == EvalMode.ON_DEVICE:
        default_output_dir = "export/on_device_e2e"
    else:  # quantsim
        default_output_dir = "export/quantsim"
    output_dir = args.output_dir or default_output_dir
    if not is_test:
        display_or_save_image(pil_img, output_dir)
