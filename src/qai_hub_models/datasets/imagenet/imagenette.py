# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

IMAGENETTE_FOLDER_NAME = "imagenette2-320"
IMAGENETTE_VERSION = 2
IMAGENETTE_FILENAME = "imagenette2-320.tgz"
IMAGENETTE_ASSET = CachedWebDatasetAsset(
    f"https://s3.amazonaws.com/fast-ai-imageclas/{IMAGENETTE_FILENAME}",
    IMAGENETTE_FOLDER_NAME,
    IMAGENETTE_VERSION,
    "imagenette2-320.tgz",
    private_s3_key=f"qai-hub-models/datasets/imagenette/{IMAGENETTE_FILENAME}",
)
