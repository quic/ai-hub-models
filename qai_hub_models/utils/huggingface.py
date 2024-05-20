# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from huggingface_hub.utils import GatedRepoError
from packaging import version

from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, ModelZooAssetConfig
from qai_hub_models.utils.base_model import TargetRuntime


def fetch_huggingface_target_model(
    model_name: str,
    dst_folder: str | Path,
    runtime_path: TargetRuntime = TargetRuntime.TFLITE,
    config: ModelZooAssetConfig = ASSET_CONFIG,
) -> List[str]:
    fs = HfFileSystem()
    hf_path = config.get_huggingface_path(model_name)

    if runtime_path == TargetRuntime.TFLITE:
        file_types = ["tflite"]
    elif runtime_path == TargetRuntime.QNN:
        file_types = ["so", "bin"]
    else:
        raise NotImplementedError()

    files = []
    for file_type in file_types:
        files += fs.glob(os.path.join(hf_path, f"**/*.{file_type}"))
    if not files:
        raise FileNotFoundError(
            f"No compiled assets are available on Huggingface for {model_name} with runtime {runtime_path.name}."
        )

    os.makedirs(dst_folder, exist_ok=True)
    paths = []
    for file in files:
        path = hf_hub_download(hf_path, file[len(hf_path) + 1 :], local_dir=dst_folder)
        paths.append(path)

    return paths


def has_model_access(repo_name: str, repo_url: str):
    # Huggingface returns GatedRepoError if model is not accessible to current User.
    # ref: https://github.com/huggingface/huggingface_hub/blob/5ff2d150d121d04799b78bc08f2343c21b8f07a9/src/huggingface_hub/utils/_errors.py#L135

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
