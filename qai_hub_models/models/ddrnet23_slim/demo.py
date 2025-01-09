# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.segmentation.demo import segmentation_demo
from qai_hub_models.models.ddrnet23_slim.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DDRNet,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.png"
)


def main(is_test: bool = False):
    segmentation_demo(DDRNet, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
