# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.super_resolution.demo import super_resolution_demo
from qai_hub_models.models.quicksrnetlarge.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    QuickSRNetLarge,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "quicksrnetlarge_demo.jpg"
)


# Run QuickSRNet end-to-end on a sample image.
# The demo will display an upscaled image
def main(is_test: bool = False):
    super_resolution_demo(
        model_cls=QuickSRNetLarge,
        default_image=IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
