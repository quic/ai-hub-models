# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import pathlib

from setuptools import setup

qaihm_path = (pathlib.Path(__file__).parent / "qai_hub_models").absolute()
qaihm_model_dir = qaihm_path / "models"
reqs_filename = "requirements.txt"


def load_requirements(path: str | os.PathLike) -> list[str]:
    """
    Read requirements from the given path, return a list of pip-parseable requirements.
    Ignore / remove comments.
    """
    with open(path) as file:
        return [
            line.split("#")[0].strip()
            for line in file
            if line.strip() and not line.startswith("#")
        ]


def get_extras() -> dict[str, list[str]]:
    """Generate the valid extras for this version of AI Hub Models."""
    extras_require = {
        "dev": [
            line.split("#")[0].strip()
            for line in open(qaihm_path / "requirements-dev.txt").readlines()
        ]
    }

    # Create extra for every model that requires one.
    for model_dir in qaihm_model_dir.iterdir():
        if not model_dir.is_file() and (model_dir / reqs_filename).exists():
            extra_with_dash = model_dir.name.replace("_", "-")
            reqs = load_requirements(model_dir / reqs_filename)
            extras_require[model_dir.name] = reqs
            extras_require[extra_with_dash] = reqs

    return extras_require


setup(
    install_requires=load_requirements(qaihm_path / "requirements.txt"),
    extras_require=get_extras(),
)
