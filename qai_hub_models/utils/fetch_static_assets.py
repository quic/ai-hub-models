# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, ModelZooAssetConfig
from qai_hub_models.utils.huggingface import fetch_huggingface_target_model
from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub
from qai_hub_models.utils.version_helpers import QAIHMVersion

try:
    from qai_hub_models.utils._internal.fetch_static_assets_internal import (
        fetch_static_assets_internal,
    )
except ImportError:
    fetch_static_assets_internal = None  # type: ignore[assignment]


def fetch_static_assets(
    model_id: str,
    runtime: TargetRuntime,
    precision: Precision = Precision.float,
    device: ScorecardDevice | None = cs_universal,
    components: list[str] | None = None,
    qaihm_version_tag: str | None = None,
    output_folder: str | os.PathLike | None = None,
    asset_config: ModelZooAssetConfig = ASSET_CONFIG,
) -> tuple[list[Path], list[str]]:
    """
    Fetch previously released assets for a model to disk, and place them in the output folder.

    Parameters:
        model_id:
            Model ID to fetch
        runtime:
            Target Runtime to fetch
        precision:
            Precision to fetch
        device:
            Device for which assets should be fetched. Ignored if runtime is not compiled for a specific device.
        components:
            Components to fetch. If None, returns all components.
        qaihm_version_tag:
            QAIHM version tag to fetch (ex. "v0.33.0"). If None, gets the assets from the currently installed QAIHM version.
        output_folder:
            If set, downloads all assets to this folder. If not set, only file URLs are returned. THe paths list will be empty.
        asset_config:
            QAIHM asset config.

    Returns:
        - A list of downloaded model file paths in order of the components list, if output_folder is set. Empty list if output_folder is None.
        - Model file URLs, in order of the model list
    """
    info = QAIHMModelInfo.from_model(model_id)
    if runtime.is_aot_compiled and device is None:
        raise ValueError(
            "You must specify a device to fetch an asset that is device-specific."
        )

    try:
        return fetch_huggingface_target_model(
            info.name,
            components,
            precision,
            device.chipset if device and runtime.is_aot_compiled else None,
            runtime,
            (
                QAIHMVersion.tag_from_string(qaihm_version_tag)
                if qaihm_version_tag
                else QAIHMVersion.current_tag
            ),
            output_folder,
            asset_config,
        )

    except FileNotFoundError as e:
        if fetch_static_assets_internal is None or not can_access_qualcomm_ai_hub():
            raise e
        print(
            "Model not found on Hugging Face. Using AI Hub to fetch the assets directly."
        )
        return fetch_static_assets_internal(
            model_id,
            runtime,
            precision,
            device,
            components,
            qaihm_version_tag,
            output_folder,
            asset_config,
        )
