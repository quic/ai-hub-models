# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import ast

import numpy as np

from qai_hub_models.models.face_det_lite.app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.demo import main as demo_main
from qai_hub_models.models.face_det_lite.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.models.face_det_lite_quantized.model import FaceDetLiteQuantizable
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_640x480_Rooney.jpg"
)
OUTPUT_RST_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "outputs.json"
)


# Verify that the output from Torch is as expected. bbox, landmark
def test_task():
    parser = get_model_cli_parser(FaceDetLiteQuantizable)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([])
    args.aimet_encodings = None  # TODO test, remove later
    model = demo_model_from_cli_args(FaceDetLiteQuantizable, MODEL_ID, args)
    app = FaceDetLiteApp(model)
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_tensor = app.run_inference_on_image(original_image)
    output_tensor_oracle = load_json(OUTPUT_RST_ADDRESS)
    bounding_box_list = ast.literal_eval(output_tensor_oracle["bounding box"])

    for i in range(len(output_tensor)):
        assert (
            np.array(output_tensor[i]) - np.array(bounding_box_list[i])
        ).mean() < 0.3


@skip_clone_repo_check
def test_demo():
    demo_main()
