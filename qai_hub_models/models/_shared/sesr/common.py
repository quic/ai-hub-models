# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.utils.asset_loaders import SourceAsRoot

# SESR original repo is here: https://github.com/ARM-software/sesr
# But this is all written in TF and Keras. Torch version is in AIMET
SESR_SOURCE_REPOSITORY = "https://github.com/quic/aimet-model-zoo"
SESR_SOURCE_REPO_COMMIT = "d09d2b0404d10f71a7640a87e9d5e5257b028802"


def _load_sesr_source_model(
    model_id, model_asset_version: int | str, scaling_factor, num_channels, num_lblocks
) -> torch.nn.Module:
    # Load SESR model from the source repository using the given weights.
    # Returns <source repository>.utils.super_resolution.models.SESRRelease
    with SourceAsRoot(
        SESR_SOURCE_REPOSITORY, SESR_SOURCE_REPO_COMMIT, model_id, model_asset_version
    ):

        from aimet_zoo_torch.common.super_resolution.models import SESRRelease

        return SESRRelease(
            scaling_factor=scaling_factor,
            num_channels=num_channels,
            num_lblocks=num_lblocks,
        )
