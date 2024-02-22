# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.super_resolution.demo import super_resolution_demo
from qai_hub_models.models.xlsr_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    XLSRQuantizable,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import TargetRuntime

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "xlsr_quantized_demo.jpg"
)


def main(is_test: bool = False):
    super_resolution_demo(
        XLSRQuantizable,
        IMAGE_ADDRESS,
        is_test,
        available_target_runtimes=[TargetRuntime.TFLITE],
    )


if __name__ == "__main__":
    main()
