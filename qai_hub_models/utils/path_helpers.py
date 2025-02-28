# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import subprocess
from pathlib import Path

from qai_hub_models._version import __version__
from qai_hub_models.utils.asset_loaders import load_yaml

MODELS_PACKAGE_NAME = "models"
QAIHM_PACKAGE_NAME = "qai_hub_models"
QAIHM_PACKAGE_ROOT = Path(__file__).parent.parent


def _get_qaihm_models_root(package_root: Path = QAIHM_PACKAGE_ROOT) -> Path:
    return package_root / MODELS_PACKAGE_NAME


QAIHM_MODELS_ROOT = _get_qaihm_models_root()


def _get_all_models(public_only: bool = False, models_root: Path = QAIHM_MODELS_ROOT):
    all_models = []
    for subdir in models_root.iterdir():
        if not subdir.is_dir():
            continue
        # Heuristic to see if this is a model we should generate export.py for.
        if (subdir / "info.yaml").exists():
            if public_only:
                if load_yaml(subdir / "info.yaml").get("status") != "public":
                    continue
            all_models.append(subdir.name)
    return all_models


MODEL_IDS = sorted(_get_all_models())


def get_git_branch():
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True
        )
        if "not a git repository" in res.stderr.decode():
            # repo not found, this must be a release
            return f"release_{__version__}"
        elif res.stderr:
            # unknown why git failed
            return f"unknown_git_branch_error__{__version__}"
        # return branch name
        return res.stdout.decode()[:-1]
    except FileNotFoundError:
        # git not found, this must be a release
        return f"release_{__version__}"
