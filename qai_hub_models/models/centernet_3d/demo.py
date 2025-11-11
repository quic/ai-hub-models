# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.centernet_3d.app import CenterNet3DApp
from qai_hub_models.models.centernet_3d.model import IMAGE, MODEL_ID, CenterNet3D
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image


def main(is_test: bool = False):
    parser = get_model_cli_parser(CenterNet3D)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    args = parser.parse_args([] if is_test else None)

    inference_model = model_from_cli_args(CenterNet3D, args)
    model = demo_model_from_cli_args(CenterNet3D, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load
    app = CenterNet3DApp(
        model,  # type: ignore[arg-type]
        inference_model.decode,
    )

    image = load_image(IMAGE.fetch())

    output_images = app.predict_3d_boxes_from_image(image)

    if not is_test:
        # visualize images
        display_or_save_image(output_images[0], args.output_dir, "centernet_bbox.png")
        display_or_save_image(output_images[1], args.output_dir, "centernet_bev.png")


if __name__ == "__main__":
    main()
