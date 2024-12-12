# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.depth_estimation.demo import depth_estimation_demo
from qai_hub_models.models.midas.model import MODEL_ASSET_VERSION, MODEL_ID, Midas
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

# Demo image comes from https://github.com/pytorch/hub/raw/master/images/dog.jpg
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.jpg"
)


def main(is_test: bool = False):
    depth_estimation_demo(Midas, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
