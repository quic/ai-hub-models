# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
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


def _get_all_models(
    public_only: bool = False, models_root: Path = QAIHM_MODELS_ROOT
) -> list[str]:
    all_models: list[str] = []
    for subdir in models_root.iterdir():
        if not subdir.is_dir():
            continue
        # Heuristic to see if this is a model we should generate export.py for.
        if (subdir / "info.yaml").exists():
            if (
                public_only
                and load_yaml(subdir / "info.yaml").get("status") != "public"
            ):
                continue
            all_models.append(subdir.name)
    return all_models


MODEL_IDS = sorted(_get_all_models())


def get_next_free_path(path: str | os.PathLike, delim: str = "-") -> Path:
    """Adds an incrementing number at the end of the given path until a non-existant filename is found."""
    path_without_ext, ext = os.path.splitext(path)

    counter = 1
    while os.path.exists(path):
        # Create new filename with underscore and counter
        path = f"{path_without_ext}{delim}{counter}{ext}"
        counter += 1

    return Path(path)


def get_git_branch() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            capture_output=True,
        )
        if "not a git repository" in res.stderr.decode():
            # repo not found, this must be a release
            return f"release_{__version__}"
        if res.stderr:
            # unknown why git failed
            return f"unknown_git_branch_error__{__version__}"
        # return branch name
        return res.stdout.decode()[:-1]
    except FileNotFoundError:
        # git not found, this must be a release
        return f"release_{__version__}"
