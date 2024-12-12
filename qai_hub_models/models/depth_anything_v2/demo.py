# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.depth_estimation.demo import depth_estimation_demo
from qai_hub_models.models.depth_anything_v2.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DepthAnythingV2,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.jpg"
)


def main(is_test: bool = False):
    depth_estimation_demo(DepthAnythingV2, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
