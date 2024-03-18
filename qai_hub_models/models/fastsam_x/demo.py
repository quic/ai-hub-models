# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.fastsam.demo import fastsam_demo
from qai_hub_models.models.fastsam_x.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FastSAM_X,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

INPUT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image_640.jpg"
)


def main(is_test: bool = False):
    fastsam_demo(FastSAM_X, MODEL_ID, image_path=INPUT_IMAGE, is_test=is_test)


if __name__ == "__main__":
    main()
