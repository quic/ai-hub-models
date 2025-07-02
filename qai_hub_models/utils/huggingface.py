# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, hf_hub_url
from huggingface_hub.utils import GatedRepoError
from packaging import version

from qai_hub_models._version import __version__
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, ModelZooAssetConfig
from qai_hub_models.utils.base_model import Precision, TargetRuntime


def get_huggingface_model_filename(
    model_name: str,
    component: str | None,
    precision: Precision,
):
    """
    Get the model file name (without the extension) that we upload to Hugging Face for the given parameters.

    Parameters:
    model_name:
        The NAME of the model (NOT THE MODEL ID)
        Typically this is QAIHMModelInfo.from_model(model_id).name

    component:
        Model component name.
        If this is None or the same string as parameter 'model_name',
        this function assumes the model has only 1 component.

    precision:
        Model precision.
    """
    precision_ext = f"_{precision}" if precision != Precision.float else ""
    if component == model_name or component is None:
        return f"{model_name}{precision_ext}"
    return f"{model_name}_{component}{precision_ext}"


def fetch_huggingface_target_model(
    model_name: str,
    model_components: list[str] | None,
    precision: Precision,
    dst_folder: str | Path,
    runtime_path: TargetRuntime = TargetRuntime.TFLITE,
    config: ModelZooAssetConfig = ASSET_CONFIG,
    qaihm_version_tag: str | None = f"v{__version__}",
    download: bool = True,
) -> tuple[list[str], list[str]]:
    fs = HfFileSystem()
    hf_path = config.get_huggingface_path(model_name)
    file_types = [runtime_path.file_extension]

    files = []
    for component_name in model_components or [None]:  # type: ignore[list-item]
        for file_type in file_types:
            files += fs.glob(
                os.path.join(
                    hf_path,
                    f"{get_huggingface_model_filename(model_name, component_name, precision)}.{file_type}",
                )
            )

    if not files:
        raise FileNotFoundError(
            f"No compiled assets are available on Huggingface for {model_name} with runtime {runtime_path.name}."
        )

    os.makedirs(dst_folder, exist_ok=True)
    paths = []
    urls = []
    for file in files:
        if download:
            path = hf_hub_download(
                hf_path,
                file[len(hf_path) + 1 :],
                local_dir=dst_folder,
                revision=qaihm_version_tag,
            )
            paths.append(path)

        url = hf_hub_url(
            hf_path,
            file[len(hf_path) + 1 :],
            revision=qaihm_version_tag,
        )
        urls.append(url)

    return paths, urls


def has_model_access(repo_name: str, repo_url: str | None = None):
    # Huggingface returns GatedRepoError if model is not accessible to current User.
    # ref: https://github.com/huggingface/huggingface_hub/blob/5ff2d150d121d04799b78bc08f2343c21b8f07a9/src/huggingface_hub/utils/_errors.py#L135

    if not repo_url:
        repo_url = "https://huggingface.co/" + repo_name

    try:
        hf_api = HfApi()
        hf_api.model_info(repo_name)
    except GatedRepoError:
        no_access_error = (
            f"Seems like you don't have access to {repo_name} yet.\nPlease follow the following steps:"
            f"\n 1. Apply for access at {repo_url}"
            f"\n 2. Setup Huggingface API token as described in https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command"
            f"\nOnce access request is approved, you should be able to export/load {repo_name} via AI-Hub."
        )
        raise RuntimeError(no_access_error)

    # Model is accesible for current User.
    return True


def ensure_has_required_transformer(least_expected_version):
    # import transformer as part of this function
    # to avoid leaking installation globally on file import.
    # NOTE: #10761 this function should not be required once AIMET (https://pypi.org/project/aimet-torch/)
    # remove tight dependency on transformers.
    import transformers

    if version.parse(transformers.__version__) < version.parse(least_expected_version):
        raise RuntimeError(
            f"Installed transformers version not supported. Expected >= {least_expected_version}, got {str(transformers.__version__)}\n"
            f"Please run `pip install transformers=={least_expected_version}`"
        )
