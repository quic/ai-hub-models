# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import subprocess
from typing import Iterable

from .constants import PY_PACKAGE_INSTALL_ROOT, PY_PACKAGE_MODELS_ROOT, REPO_ROOT
from .task import CompositeTask, RunCommandsTask, RunCommandsWithVenvTask
from .util import can_support_aimet, model_needs_aimet


class CreateVenvTask(RunCommandsTask):
    def __init__(self, venv_path: str, python_executable: str) -> None:
        super().__init__(
            f"Creating virtual environment at {venv_path}",
            f"source {REPO_ROOT}/scripts/util/env_create.sh --python={python_executable} --venv={venv_path} --no-sync",
        )


def is_package_installed(package_name: str, venv_path: str | None = None) -> bool:
    if venv_path is not None:
        if not os.path.exists(venv_path):
            return False
        command = f'. {venv_path}/bin/activate && python -c "import {package_name}"'
    else:
        command = f'python -c "import {package_name}"'

    try:
        subprocess.check_call(command, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


class SyncLocalQAIHMVenvTask(CompositeTask):
    """Sync the provided environment with local QAIHM and the provided extras."""

    def __init__(
        self,
        venv_path: str | None,
        extras: Iterable[str] = [],
        include_aimet: bool = can_support_aimet(),
    ) -> None:
        tasks = []
        extras_str = f"[{','.join(extras)}]" if extras else ""
        tasks.append(
            RunCommandsWithVenvTask(
                group_name=f"Install QAIHM{extras_str}",
                venv=venv_path,
                commands=[
                    f'pip install -e "{PY_PACKAGE_INSTALL_ROOT}{extras_str}" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html',
                ],
            )
        )

        super().__init__(
            f"Create Local QAIHM{extras_str} Virtual Environment at {venv_path}",
            [task for task in tasks],
        )


class SyncModelVenvTask(SyncLocalQAIHMVenvTask):
    """Sync the provided environment with local QAIHM and the provided extras needed for the model_name."""

    def __init__(
        self,
        model_name,
        venv_path,
        include_dev_deps: bool = False,
        only_model_requirements: bool = False,
    ) -> None:
        extras = []
        if include_dev_deps:
            extras.append("dev")
        if os.path.exists(
            os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "requirements.txt")
        ):
            extras.append(model_name)

        super().__init__(
            venv_path,
            extras,
            model_needs_aimet(model_name),
        )


class SyncModelRequirementsVenvTask(RunCommandsWithVenvTask):
    """Sync the provided environment with requirements from model_name's requirements.txt.
    Will not re-install QAI Hub Models. Intended for speeding up CI compared to building an entirely new env for each model."""

    def __init__(self, model_name, venv_path, pip_force_install: bool = True) -> None:
        requirements_txt = os.path.join(
            PY_PACKAGE_MODELS_ROOT, model_name, "requirements.txt"
        )
        if os.path.exists(requirements_txt):
            commands = [
                f'pip install {"--force-reinstall" if pip_force_install else None} -r "{requirements_txt}" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html'
            ]
        else:
            commands = []

        super().__init__(
            group_name=f"Install Model Requirements for {model_name}",
            venv=venv_path,
            commands=commands,
        )
