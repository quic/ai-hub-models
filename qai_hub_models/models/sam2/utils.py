# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import shutil
from pathlib import Path


def copy_configs(src: str, dst: Path) -> None:
    """
    Copies a configuration directory from the source path to a destination path,
    ensuring it is placed within the required subdirectory structure.

    If the destination path does not already include the required subdirectory
    'qai_hub_models/models/sam2', it appends this structure along with
    'configs/sam2.1' to the destination.

    Args:
        src (str): Path to the source configuration directory.
        dst (Path): Base path where the configuration directory should be copied.

    Returns:
        None
    """

    src_path = Path(src)
    dst_path = Path(dst) / "configs" / src_path.name
    required_subdir = "qai_hub_models/models/sam2"
    if required_subdir not in str(dst_path):
        dst_path = dst_path = Path(dst) / required_subdir / "configs" / "sam2.1"

    # Copy the entire directory tree
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
