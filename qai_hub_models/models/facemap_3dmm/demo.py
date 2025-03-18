# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import cv2
import numpy as np
from PIL import Image
from skimage import io

from qai_hub_models.models.facemap_3dmm.app import FaceMap_3DMMApp
from qai_hub_models.models.facemap_3dmm.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceMap_3DMM,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.display import display_or_save_image

INPUT_IMAGE_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "face_img.jpg"
)
INPUT_FBOX_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "face_img_fbox.txt"
)


# Run FaceMap_3DMM end-to-end on a sample image.
# The demo will display a image with the predicted landmark displayed.
def facemap_3dmm_demo(
    model_cls: type[FaceMap_3DMM] = FaceMap_3DMM,
    model_id: str = MODEL_ID,
    is_test: bool = False,
):
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=str(INPUT_IMAGE_PATH.fetch()),
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_cls, model_id, args)
    validate_on_device_demo_args(args, model_id)

    # Load image
    image = io.imread(args.image)

    print("Model Loaded")

    app = FaceMap_3DMMApp(model)

    # Get face bounding box info (from file or face detector)
    fbox = np.loadtxt(INPUT_FBOX_PATH.fetch())
    x0, x1, y0, y1 = (
        np.int32(fbox[0]),
        np.int32(fbox[1]),
        np.int32(fbox[2]),
        np.int32(fbox[3]),
    )

    lmk, output = app.landmark_prediction(image, x0, x1, y0, y1)

    if not is_test:
        # Annotated lmk
        np.savetxt(
            "build/demo_output_lmk.txt",
            lmk.detach().numpy(),
        )

        # Annotated image
        display_or_save_image(
            Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)),
            filename="demo_output_img.png",
        )


if __name__ == "__main__":
    facemap_3dmm_demo(FaceMap_3DMM, MODEL_ID)
