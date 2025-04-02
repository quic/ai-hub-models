# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.segmentation.demo import segmentation_demo
from qai_hub_models.models.unet_segmentation.app import UNetSegmentationApp
from qai_hub_models.models.unet_segmentation.model import IMAGE_ADDRESS, MODEL_ID, UNet


def main(is_test: bool = False):
    # This model is sensitive to constant-value padding
    pad_mode = "reflect"
    segmentation_demo(
        UNet,
        MODEL_ID,
        IMAGE_ADDRESS,
        is_test,
        pad_mode=pad_mode,
        app_cls=UNetSegmentationApp,
    )


if __name__ == "__main__":
    main()
