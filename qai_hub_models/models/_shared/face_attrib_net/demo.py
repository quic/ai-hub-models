# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json
from pathlib import Path

from qai_hub_models.models._shared.face_attrib_net.app import FaceAttribNetApp
from qai_hub_models.models.face_attrib_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OUT_NAMES,
    FaceAttribNet,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "img_sample.bmp"
)


# Run FaceAttribNet end-to-end on a sample image.
def face_attrib_net_demo(model_cls: type[FaceAttribNet], is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([])
    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image
    _, _, height, width = model_cls.get_input_spec()["image"][0]
    orig_image = load_image(args.image)
    print("Model loaded")

    app = FaceAttribNetApp(model)
    output = app.run_inference_on_image(orig_image)
    out_dict = {}
    for i in range(len(output)):
        out_dict[OUT_NAMES[i]] = list(output[i].astype(float))

    output_path = (args.output_dir or str(Path() / "build")) + "/output.json"
    with open(output_path, "w", encoding="utf-8") as wf:
        json.dump(out_dict, wf, ensure_ascii=False, indent=4)
    print(f"Model outputs are saved at: {output_path}")
