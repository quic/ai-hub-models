# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.super_resolution.demo import super_resolution_demo
from qai_hub_models.models.real_esrgan_x4plus.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Real_ESRGAN_x4plus,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "real_esrgan_x4plus_demo.jpg"
)
WEIGHTS_HELP_MSG = "RealESRGAN checkpoint `.pth` name from the Real-ESRGAN repo. Can be set to any of the model names defined here: https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md to automatically download the file instead."


# Run Real-ESRGAN end-to-end on a sample image.
# The demo will display a image with the predicted bounding boxes.
def main(is_test: bool = False):
    super_resolution_demo(
        model_cls=Real_ESRGAN_x4plus,
        model_id=MODEL_ID,
        default_image=IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
