# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo

SOURCE_REPO = "https://github.com/NVIDIA/DeepLearningExamples"
COMMIT_HASH = "9dd9fcb98f56187e49c5ee280cf8dbd530dde57b"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class GPUNet(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls):
        with SourceAsRoot(
            SOURCE_REPO, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION
        ) as repo_path:
            gpu_net_repo_path = (
                Path(repo_path) / "PyTorch" / "Classification" / "GPUNet"
            )
            find_replace_in_repo(
                str(gpu_net_repo_path),
                "configs/model_hub.py",
                "if download:",
                "if download and not(Path(checkpointPath).exists()):",
            )
            sys.path.append(str(gpu_net_repo_path))

            from configs.model_hub import get_configs
            from models.gpunet_builder import GPUNet_Builder
            from timm.models.helpers import load_checkpoint

            modelJSON, ckpt_path = get_configs(
                batch=1,
                latency="0.65ms",
                gpuType="GV100",
                config_root_dir=str(gpu_net_repo_path / "configs"),
            )
            builder = GPUNet_Builder()
            model = builder.get_model(modelJSON)
            load_checkpoint(model, ckpt_path)

            return cls(model).eval()
