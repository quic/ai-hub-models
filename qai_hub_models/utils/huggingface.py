# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from huggingface_hub import HfFileSystem, hf_hub_download

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
