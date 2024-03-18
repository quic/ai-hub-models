# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from pathlib import Path
from typing import Optional

from qai_hub_models.utils.asset_loaders import load_yaml

MODELS_PACKAGE_NAME = "models"
QAIHM_PACKAGE_NAME = "qai_hub_models"


def get_all_models(public_only: bool = False):
    zoo_root = get_qaihm_models_root()
    all_models = []
    for subdir in zoo_root.iterdir():
        if not subdir.is_dir():
            continue
        # Heuristic to see if this is a model we should generate export.py for.
        if (subdir / "model.py").exists() and (subdir / "test.py").exists():
            if public_only:
                if not (subdir / "info.yaml").exists():
                    continue
                if load_yaml(subdir / "info.yaml").get("status") != "public":
                    continue
            all_models.append(subdir.name)
    return all_models


def get_qaihm_package_root() -> Path:
    """Get local path to qaihm package root."""
    return Path(__file__).parent.parent


def get_qaihm_models_root(package_root: Optional[Path] = None) -> Path:
    if package_root is None:
        package_root = get_qaihm_package_root()
    return package_root / MODELS_PACKAGE_NAME
