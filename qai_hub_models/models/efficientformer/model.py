# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
    load_torch,
)

SOURCE_REPO = "https://github.com/snap-research/EfficientFormer"
COMMIT_HASH = "d01da426243a38303d0a35d01fc928b0e37d38d3"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# originally from https://drive.google.com/file/d/1wtEmkshLFEYFsX5YhBttBOGYaRvDR7nu/view?usp=sharing
DEFAULT_WEIGHTS = "efficientformer_l1_300d"
DEFAULT_WEIGHTS_FILE = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    f"{DEFAULT_WEIGHTS}.pth",
)


class EfficientFormer(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS):
        weights_file = weights_name
        if weights_name == DEFAULT_WEIGHTS:
            weights_file = DEFAULT_WEIGHTS_FILE

        weights = load_torch(weights_file)["model"]

        with SourceAsRoot(
            SOURCE_REPO, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION
        ) as repo_path:
            find_replace_in_repo(
                repo_path,
                ["models/efficientformer.py", "models/efficientformer_v2.py"],
                "timm.models.layers.helpers",
                "timm.layers.helpers",
            )
            from models.efficientformer import efficientformer_l1

            model = efficientformer_l1()
            model.load_state_dict(weights)
            return cls(model).eval()
