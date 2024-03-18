# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch

from qai_hub_models.utils.aimet.repo import aimet_zoo_as_root


def _load_quicksrnet_source_model(
    scaling_factor,
    num_channels,
    num_intermediate_layers,
    use_ito_connection,
) -> torch.nn.Module:
    # Load QuickSRNet model from the source repository using the given weights.
    # Returns <source repository>.utils.super_resolution.models.QuickSRNetBase
    with aimet_zoo_as_root():
        from aimet_zoo_torch.quicksrnet.model.models import QuickSRNetBase

        return QuickSRNetBase(
            scaling_factor=scaling_factor,
            num_channels=num_channels,
            num_intermediate_layers=num_intermediate_layers,
            use_ito_connection=use_ito_connection,
        )
