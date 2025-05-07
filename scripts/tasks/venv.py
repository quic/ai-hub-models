# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable

from .constants import PY_PACKAGE_INSTALL_ROOT, PY_PACKAGE_MODELS_ROOT, REPO_ROOT
from .task import RunCommandsTask, RunCommandsWithVenvTask
from .util import get_code_gen_str_field, get_pip


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


class GenerateGlobalRequirementsTask(RunCommandsWithVenvTask):
    # Global requirements change based on the python version,
    # and should therefore be regenerated before running any model tests.
    def __init__(
        self,
        venv,
        env=None,
        raise_on_failure=True,
        ignore_return_codes: list[int] | None = None,
    ):
        super().__init__(
            "Generate Global Requirements",
            venv,
            ["python -m qai_hub_models.scripts.generate_global_requirements"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class AggregateScorecardResultsTask(RunCommandsWithVenvTask):
    def __init__(
        self,
        venv,
        env=None,
        raise_on_failure=True,
        ignore_return_codes: list[int] | None = None,
    ):
        super().__init__(
            "Aggregate Scorecard Results",
            venv,
            ["python -m qai_hub_models.scripts.aggregate_scorecard_results"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class DownloadPrivateDatasetsTask(RunCommandsWithVenvTask):
    # Needed to quantize models relying on data without public download links
    def __init__(
        self,
        venv,
        env=None,
        raise_on_failure=True,
        ignore_return_codes: list[int] | None = None,
    ):
        super().__init__(
            "Download Private Datasets",
            venv,
            ["python -m qai_hub_models.scripts.download_private_datasets"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class SyncLocalQAIHMVenvTask(RunCommandsWithVenvTask):
    """Sync the provided environment with local QAIHM and the provided extras."""

    def __init__(
        self,
        venv_path: str | None,
        extras: Iterable[str] = [],
        flags: str | None = None,
        pre_install: str | None = None,
    ) -> None:
        extras_str = f"[{','.join(extras)}]" if extras else ""
        commands = [
            f'{get_pip()} install -e "{PY_PACKAGE_INSTALL_ROOT}{extras_str}" '
            + (flags or "")
        ]
        if pre_install:
            commands.insert(0, f"{get_pip()} install {pre_install}")

        super().__init__(
            group_name=f"Install QAIHM{extras_str}", venv=venv_path, commands=commands
        )


class SyncModelVenvTask(SyncLocalQAIHMVenvTask):
    """Sync the provided environment with local QAIHM and the provided extras needed for the model_name."""

    def __init__(
        self,
        model_name,
        venv_path,
        include_dev_deps: bool = False,
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
            get_code_gen_str_field(model_name, "pip_install_flags"),
            get_code_gen_str_field(model_name, "pip_pre_build_reqs"),
        )


class SyncModelRequirementsVenvTask(RunCommandsWithVenvTask):
    """Sync the provided environment with requirements from model_name's requirements.txt.
    Will not re-install QAI Hub Models. Intended for speeding up CI compared to building an entirely new env for each model."""

    def __init__(self, model_name, venv_path, pip_force_install: bool = True) -> None:
        requirements_txt = os.path.join(
            PY_PACKAGE_MODELS_ROOT, model_name, "requirements.txt"
        )
        extra_flags = get_code_gen_str_field(model_name, "pip_install_flags")
        pre_install = get_code_gen_str_field(model_name, "pip_pre_build_reqs")
        if os.path.exists(requirements_txt):
            commands = [
                f'{get_pip()} install {"--force-reinstall" if pip_force_install else None} -r "{requirements_txt}" {extra_flags or ""}'
            ]
            if pre_install:
                commands.insert(0, f"{get_pip()} install {pre_install}")
        else:
            commands = []

        super().__init__(
            group_name=f"Install Model Requirements for {model_name}",
            venv=venv_path,
            commands=commands,
        )
