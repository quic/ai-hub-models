# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch

from qai_hub_models.utils.asset_loaders import SourceAsRoot

QUICKSRNET_SOURCE_REPOSITORY = "https://github.com/quic/aimet-model-zoo"
QUICKSRNET_SOURCE_REPO_COMMIT = "d09d2b0404d10f71a7640a87e9d5e5257b028802"


def _load_quicksrnet_source_model(
    model_id,
    model_asset_version,
    scaling_factor,
    num_channels,
    num_intermediate_layers,
    use_ito_connection,
) -> torch.nn.Module:
    # Load QuickSRNet model from the source repository using the given weights.
    # Returns <source repository>.utils.super_resolution.models.QuickSRNetBase
    with SourceAsRoot(
        QUICKSRNET_SOURCE_REPOSITORY,
        QUICKSRNET_SOURCE_REPO_COMMIT,
        model_id,
        model_asset_version,
    ):
        # Remove import of model_definition.py as it has an import error itself,
        # but we don't need anything from that file here
        with open("aimet_zoo_torch/quicksrnet/__init__.py", "r") as file:
            file_content = file.read()
        new_content = file_content.replace(
            "from .model.model_definition import QuickSRNet", " "
        )
        with open("aimet_zoo_torch/quicksrnet/__init__.py", "w") as file:
            file.write(new_content)

        from aimet_zoo_torch.quicksrnet.model.models import QuickSRNetBase

        return QuickSRNetBase(
            scaling_factor=scaling_factor,
            num_channels=num_channels,
            num_intermediate_layers=num_intermediate_layers,
            use_ito_connection=use_ito_connection,
        )
