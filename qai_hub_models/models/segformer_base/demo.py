# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.segmentation.demo import segmentation_demo
from qai_hub_models.models.segformer_base.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SegformerBase,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "out_image_512_with_mask.png"
)


def main(is_test: bool = False):
    segmentation_demo(SegformerBase, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
