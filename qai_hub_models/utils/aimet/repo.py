# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from contextlib import contextmanager

from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo

AIMET_ZOO_SOURCE_REPOSITORY = "https://github.com/quic/aimet-model-zoo"
AIMET_ZOO_SOURCE_REPO_COMMIT = "d09d2b0404d10f71a7640a87e9d5e5257b028802"
REPO_ASSET_VERSION = 1


@contextmanager
def aimet_zoo_as_root():
    with SourceAsRoot(
        AIMET_ZOO_SOURCE_REPOSITORY,
        AIMET_ZOO_SOURCE_REPO_COMMIT,
        source_repo_name="aimet_zoo",
        source_repo_version=REPO_ASSET_VERSION,
        keep_sys_modules=True,
    ) as repo_root:
        # Remove import of model_definition.py as it has an import error itself,
        # but we don't need anything from that file here
        find_replace_in_repo(
            repo_root,
            "aimet_zoo_torch/quicksrnet/__init__.py",
            "from .model.model_definition import QuickSRNet",
            " ",
        )

        yield repo_root
