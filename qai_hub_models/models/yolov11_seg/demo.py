# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.yolo.demo import yolo_segmentation_demo
from qai_hub_models.models.yolov11_seg.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV11Segmentor,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/bus.jpg"
)
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/out_bus_with_mask.png"
)


def main(is_test: bool = False):
    yolo_segmentation_demo(
        YoloV11Segmentor,
        MODEL_ID,
        IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
