#!/usr/bin/env python
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import glob
import pathlib

from setuptools import find_packages, setup

r_file = "requirements.txt"

qaihm_path = pathlib.Path(__file__).parent / "qai_hub_models"
qaihm_dir = qaihm_path / "models"
requirements_path = qaihm_path / r_file

version_path = qaihm_path / "_version.py"
version_locals: dict[str, str] = {}
exec(open(version_path).read(), version_locals)


# Non-Python Files to Add to the Wheel
data_file_extensions = ["yaml", "txt", "json", "diff"]


def get_data_files() -> list[str]:
    data_files = []
    for ext in data_file_extensions:
        data_files.extend(
            glob.glob(
                f"{str(qaihm_path.absolute() / '**' / f'*.{ext}')}", recursive=True
            )
        )
    for i in range(0, len(data_files)):
        data_files[i] = data_files[i].split("qai_hub_models")[1][1:]
    return data_files


# Extras dictionary definition.
extras_require = {
    "dev": [
        line.split("#")[0].strip()
        for line in open(qaihm_path / "requirements-dev.txt").readlines()
    ]
}

# Create extra for every model that requires one.
for model_dir in qaihm_dir.iterdir():
    if not model_dir.is_file() and (model_dir / r_file).exists():
        extra_with_dash = model_dir.name.replace("_", "-")
        reqs = [
            line.split("#")[0].strip() for line in open(model_dir / r_file).readlines()
        ]
        extras_require[model_dir.name] = reqs
        extras_require[extra_with_dash] = reqs


description = "Models optimized for export to run on device."
long_description = (pathlib.Path(__file__).parent / "README.md").read_text()
setup(
    name="qai_hub_models",
    version=version_locals["__version__"],
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Qualcomm® Technologies, Inc.",
    url="https://github.com/quic/ai-hub-models",
    packages=find_packages(),
    python_requires=">=3.9, <3.13",
    package_data={"qai_hub_models": get_data_files()},
    include_package_data=True,
    install_requires=[
        line.split("#")[0].strip() for line in open(requirements_path).readlines()
    ],
    extras_require=extras_require,
    license="BSD-3",
)
