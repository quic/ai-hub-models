# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.rtmdet.app import RTMDetApp
from qai_hub_models.models.rtmdet.model import MODEL_ASSET_VERSION, MODEL_ID, RTMDet
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "rtmdet_demo_640.jpg"
)
print(IMAGE_ADDRESS)


def main(is_test: bool = False):
    yolo_detection_demo(
        RTMDet,
        MODEL_ID,
        RTMDetApp,
        IMAGE_ADDRESS,
        RTMDet.STRIDE_MULTIPLE,
        is_test=is_test,
        default_score_threshold=0.5,
    )


if __name__ == "__main__":
    main()
