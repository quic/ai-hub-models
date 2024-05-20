# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch

from qai_hub_models.models.stylegan2.app import StyleGAN2App
from qai_hub_models.models.stylegan2.model import MODEL_ID, StyleGAN2
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.display import display_or_save_image


def main(is_test: bool = False):
    parser = get_model_cli_parser(StyleGAN2)
    parser = get_on_device_demo_parser(
        parser, available_target_runtimes=[TargetRuntime.TFLITE], add_output_dir=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use for image generation.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate (all computed in one inference call).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Class[es] to use for image generation (if applicable).",
    )
    args = parser.parse_args([] if is_test else None)

    # Create model and app
    model = model_from_cli_args(StyleGAN2, args)
    inference_model = demo_model_from_cli_args(StyleGAN2, MODEL_ID, args)
    app = StyleGAN2App(inference_model, model.output_size, model.num_classes)

    # Verify model input args
    if model.num_classes == 0 and args.classes:
        raise ValueError(
            "Classes cannot be provided for models trained without classes."
        )
    if args.classes and len(args.classes) > 1 and len(args.classes) != args.num_images:
        raise ValueError(
            "You may provide 1 class for all images, or one class per image."
        )
    if not args.classes and model.num_classes:
        args.classes = [0]  # Default to class 0

    # Get desired batch size
    batch_size = len(args.classes) if args.classes else args.num_images

    # Generate input and run inference
    z = app.generate_random_vec(batch_size=batch_size, seed=args.seed)
    images = app.generate_images(
        z,
        class_idx=torch.Tensor(args.classes).type(torch.int) if args.classes else None,
    )

    # Display images
    assert isinstance(images, list)
    if not is_test:
        for (i, image) in enumerate(images):
            display_or_save_image(image, args.output_dir, f"image_{i}.png")


if __name__ == "__main__":
    main()
