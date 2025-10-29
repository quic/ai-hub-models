# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.bevfusion_det.app import BEVFusionApp
from qai_hub_models.models.bevfusion_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    BEVFusion,
)
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)
from qai_hub_models.utils.display import display_or_save_image

# Asset definitions
CAMERAS = {
    "CAM_FRONT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "images/CAM_FRONT.jpg"
    ),
    "CAM_FRONT_RIGHT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "images/CAM_FRONT_RIGHT.jpg"
    ),
    "CAM_FRONT_LEFT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "images/CAM_FRONT_LEFT.jpg"
    ),
    "CAM_BACK": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "images/CAM_BACK.jpg"
    ),
    "CAM_BACK_LEFT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "images/CAM_BACK_LEFT.jpg"
    ),
    "CAM_BACK_RIGHT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "images/CAM_BACK_RIGHT.jpg"
    ),
}
INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "inputs/inputs.json"
)


def main(is_test: bool = False):
    # Parse arguments
    parser = get_model_cli_parser(BEVFusion)

    parser = get_on_device_demo_parser(parser)

    add_output_dir_arg(parser)
    args = parser.parse_args([] if is_test else None)

    # Load model
    model = model_from_cli_args(BEVFusion, args)
    validate_on_device_demo_args(args, MODEL_ID)
    app = BEVFusionApp(
        model.encoder1,
        model.encoder2,
        model.encoder3,
        model.encoder4,
        model.decoder,
    )

    # Load inputs
    cam_paths = dict(CAMERAS.items())
    images = [load_image(str(img.fetch())) for img in cam_paths.values()]
    inputs_json = load_json(INPUTS.fetch())

    # predict
    output_images = app.predict_3d_boxes_from_images(
        images,
        cam_paths,
        inputs_json,
    )
    if not is_test:
        # visualize images
        for i, cam_name in enumerate(cam_paths):
            display_or_save_image(
                output_images[i], args.output_dir, f"bevfusion_{cam_name}.png"
            )


if __name__ == "__main__":
    main()
