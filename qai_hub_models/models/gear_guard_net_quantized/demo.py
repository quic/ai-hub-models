# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.body_detection.app import BodyDetectionApp
from qai_hub_models.models._shared.body_detection.demo import BodyDetectionDemo
from qai_hub_models.models.gear_guard_net.model import MODEL_ASSET_VERSION
from qai_hub_models.models.gear_guard_net_quantized.model import (
    MODEL_ID,
    GearGuardNetQuantizable,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    "gear_guard_net", MODEL_ASSET_VERSION, "test_image.jpg"
)


def main(is_test: bool = False):
    BodyDetectionDemo(
        is_test,
        GearGuardNetQuantizable,
        MODEL_ID,
        BodyDetectionApp,
        INPUT_IMAGE_ADDRESS,
        320,
        192,
        0.9,
    )


if __name__ == "__main__":
    main()
