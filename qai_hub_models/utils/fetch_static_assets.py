# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import qai_hub as hub
from pydantic import ValidationError

from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    ModelZooAssetConfig,
    download_file,
)
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
    device: hub.Device | None = None,
    components: list[str] | None = None,
    qaihm_version_tag: str | None = None,
    output_folder: str | os.PathLike | None = None,
    asset_config: ModelZooAssetConfig = ASSET_CONFIG,
) -> tuple[list[Path], list[str]]:
    """
    Fetch previously released assets for a model to disk, and place them in the output folder.

    Parameters
    ----------
    model_id
        Model ID to fetch.
    runtime
        Target Runtime to fetch.
    precision
        Precision to fetch.
    device
        Device for which assets should be fetched. Ignored if runtime is not compiled for a specific device.
    components
        Components to fetch. If None, returns all components.
    qaihm_version_tag
        QAIHM version tag to fetch (ex. "v0.33.0"). If None, gets the assets from the currently installed QAIHM version.
    output_folder
        If set, downloads all assets to this folder. If not set, only file URLs are returned. The paths list will be empty.
    asset_config
        QAIHM asset config.

    Returns
    -------
    paths
        A list of downloaded model file paths in order of the components list, if output_folder is set. Empty list if output_folder is None.
    urls
        Model file URLs, in order of the model list.
    """
    if runtime.is_aot_compiled and device is None:
        raise ValueError(
            "You must specify a device to fetch an asset that is device-specific."
        )

    qaihm_version_tag = (
        QAIHMVersion.tag_from_string(qaihm_version_tag)
        if qaihm_version_tag
        else QAIHMVersion.current_tag
    )

    # Load info.yaml / perf.yaml
    if qaihm_version_tag != QAIHMVersion.current_tag:
        info = _load_info_from_previous_release(
            model_id, qaihm_version_tag, asset_config
        )
    else:
        info = QAIHMModelInfo.from_model(model_id)

    chipset = None
    device_name = None
    if device is not None:
        # load devices_and_chipsets.yaml
        if qaihm_version_tag != QAIHMVersion.current_tag:
            chipsets = _load_chipsets_from_previous_release(
                qaihm_version_tag, asset_config
            )
        else:
            chipsets = DevicesAndChipsetsYaml.load()
        device_name, device_details = chipsets.get_device_details_without_aihub(device)
        chipset = device_details.chipset

    try:
        return fetch_huggingface_target_model(
            info.name,
            components,
            precision,
            chipset if chipset and runtime.is_aot_compiled else None,
            runtime,
            (
                QAIHMVersion.tag_from_string(qaihm_version_tag)
                if qaihm_version_tag
                else QAIHMVersion.current_tag
            ),
            output_folder,
            asset_config,
        )

    except FileNotFoundError:
        if fetch_static_assets_internal is None or not can_access_qualcomm_ai_hub():
            raise
        print(
            "Model not found on Hugging Face. Using AI Hub Models to fetch the assets directly."
        )
        return fetch_static_assets_internal(
            model_id,
            runtime,
            precision,
            ScorecardDevice.get(device_name) if device_name else None,
            components,
            qaihm_version_tag,
            output_folder,
            asset_config,
        )


def _load_info_from_previous_release(
    model_id: str, qaihm_version_tag: str, asset_config: ModelZooAssetConfig
) -> QAIHMModelInfo:
    """
    Fetch older info.yaml from GitHub.com, attempt to load & return it.

    Parameters
    ----------
    model_id:
        Model ID to fetch
    qaihm_version_tag:
        QAIHM version tag to fetch (ex. "v0.33.0")
    asset_config:
        QAIHM asset config.

    Returns
    -------
    QAIHMModelInfo
        Model info object.

    Raises
    ------
    FileNotFoundError
      - the yaml is unavailable on GitHub
      - the yaml schema has changed and can't be read by the current AI Hub Models version
    """
    info_url = asset_config.get_qaihm_repo_download_url(
        model_id, "info.yaml", qaihm_version_tag
    )
    with TemporaryDirectory() as tmpdir:
        try:
            info_path = download_file(
                info_url, os.path.join(tmpdir, "info.yaml"), False
            )
        except ValueError:
            raise FileNotFoundError(
                f"Unable to find asset data for {model_id} in QAIHM {qaihm_version_tag}. Verify {qaihm_version_tag} is a valid tag at {asset_config.repo_url}, and verify that the model is available in that version."
            ) from None
        try:
            return QAIHMModelInfo.from_yaml(info_path)
        except (ValueError, ValidationError):
            # The schema may have changed if we can't load the old yamls.
            raise FileNotFoundError(
                f"Unable to parse asset data for {model_id} in QAIHM {qaihm_version_tag}. Downgrade your AI Hub Models install to {qaihm_version_tag} and try again."
            ) from None


def _load_chipsets_from_previous_release(
    qaihm_version_tag: str, asset_config: ModelZooAssetConfig
) -> DevicesAndChipsetsYaml:
    """
    Fetch older devices_and_chipsets.yaml from GitHub.com, attempt to load & return it.

    Parameters
    ----------
    qaihm_version_tag:
        QAIHM version tag to fetch (ex. "v0.33.0")
    asset_config:
        QAIHM asset config.

    Returns
    -------
    DevicesAndChipsetsYaml
        Chipsets object.

    Raises
    ------
    FileNotFoundError
      - the yaml is unavailable on GitHub
      - the yaml schema has changed and can't be read by the current AI Hub Models version
    """
    chipsets_url = asset_config.get_qaihm_repo_download_url(
        None, "devices_and_chipsets.yaml", qaihm_version_tag
    )
    with TemporaryDirectory() as tmpdir:
        try:
            chipsets_path = download_file(
                chipsets_url, os.path.join(tmpdir, "devices_and_chipsets.yaml"), False
            )
        except ValueError:
            raise FileNotFoundError(
                f"Unable to find supported devices and chipsets for QAIHM {qaihm_version_tag}. Verify {qaihm_version_tag} is a valid tag at {asset_config.repo_url}, and verify that the model is available in that version."
            ) from None
        try:
            return DevicesAndChipsetsYaml.from_yaml(chipsets_path)
        except (ValueError, ValidationError):
            # The schema may have changed if we can't load the old yamls.
            raise FileNotFoundError(
                f"Unable to parse supported devices and chipsets for QAIHM {qaihm_version_tag}. Downgrade your AI Hub Models install to {qaihm_version_tag} and try again."
            ) from None
