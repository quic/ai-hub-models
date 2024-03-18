# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.utils.aimet.repo import aimet_zoo_as_root

# SESR original repo is here: https://github.com/ARM-software/sesr
# But this is all written in TF and Keras. Torch version is in AIMET Zoo


def _load_sesr_source_model(
    scaling_factor, num_channels, num_lblocks
) -> torch.nn.Module:
    # Load SESR model from the source repository using the given weights.
    # Returns <source repository>.utils.super_resolution.models.SESRRelease
    with aimet_zoo_as_root():

        from aimet_zoo_torch.common.super_resolution.models import SESRRelease

        return SESRRelease(
            scaling_factor=scaling_factor,
            num_channels=num_channels,
            num_lblocks=num_lblocks,
        )
