# ---------------------------------------------------------------------
# Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.detr.demo import detr_demo
from qai_hub_models.models.deformable_detr.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DeformableDETR,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "detr_demo_image.jpg"
)


# Run DETR app end-to-end on a sample image.
# The demo will display the predicted mask in a window.
def main(is_test: bool = False):
    detr_demo(
        DeformableDETR,
        MODEL_ID,
        DEFAULT_WEIGHTS,
        IMAGE_ADDRESS,
        is_test,
        threshold=0.75,
    )


if __name__ == "__main__":
    main()
