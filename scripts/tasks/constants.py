# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import shutil
import subprocess
import sys
from pathlib import Path


def process_output(command: subprocess.CompletedProcess[bytes]) -> str:
    return command.stdout.decode("utf-8").strip()


ON_WINDOWS = sys.platform == "win32"

DEFAULT_PYTHON = "py -3.12" if ON_WINDOWS else "python3.12"


def _find_bash() -> str | None:
    if ON_WINDOWS and (git := shutil.which("git")):
        # PATH's bash.exe is usually WSL's; we need Git Bash, which ships next to git.
        git_bash = Path(git).parent.parent / "bin" / "bash.exe"
        if git_bash.exists():
            return str(git_bash)
    return shutil.which("bash")


BASH_EXECUTABLE = _find_bash()


def bash_argv(command: str) -> list[str]:
    # shell=True + executable=bash still passes cmd.exe's "/c" flag on Windows.
    assert BASH_EXECUTABLE is not None, "bash is required (install Git for Windows)."
    return [BASH_EXECUTABLE, "-c", command]


def venv_activate_command(venv: str) -> str:
    subdir = "Scripts" if ON_WINDOWS else "bin"
    # Unset so activate's "deactivate" doesn't restore a Windows PATH inherited from an active venv.
    return f'unset _OLD_VIRTUAL_PATH; source "{venv.replace(os.sep, "/")}/{subdir}/activate"'


def run_and_get_output(command: str, check: bool = True) -> str:
    return process_output(
        subprocess.run(bash_argv(command), stdout=subprocess.PIPE, check=check)
    )


# Env Variable
STORE_ROOT_ENV_VAR = "QAIHM_STORE_ROOT"

# Set by PR compile tests to restrict split LLMs to a single instantiation.
# Must match CompileSingleInstantiationEnvvar.VARNAME in
# qai_hub_models.scorecard.envvars.
COMPILE_SINGLE_INSTANTIATION_ENV_VAR = "QAIHM_TEST_COMPILE_SINGLE_INSTANTIATION"

# Repository
REPO_ROOT = str(Path(__file__).parent.parent.parent)
VENV_PATH = os.path.join(REPO_ROOT, "qaihm-dev")
BUILD_ROOT = os.path.join(REPO_ROOT, "build")

# Dependent Wheels
QAI_HUB_LATEST_PATH = os.path.join(BUILD_ROOT, "qai_hub-latest-py3-none-any.whl")

# Package paths relative to repository root
PY_PACKAGE_RELATIVE_SRC_ROOT = os.path.join("src", "qai_hub_models")
PY_PACKAGE_RELATIVE_MODELS_ROOT = os.path.join(PY_PACKAGE_RELATIVE_SRC_ROOT, "models")

# Absolute package paths
PY_PACKAGE_INSTALL_ROOT = os.path.join(REPO_ROOT, "src")
PY_CLI_INSTALL_ROOT = os.path.join(REPO_ROOT, "cli")
PY_PACKAGE_SRC_ROOT = os.path.join(REPO_ROOT, PY_PACKAGE_RELATIVE_SRC_ROOT)
PY_CLI_SRC_ROOT = os.path.join(PY_CLI_INSTALL_ROOT, "qai_hub_models_cli")
PY_PACKAGE_LOCAL_CACHE = os.environ.get(
    STORE_ROOT_ENV_VAR, os.path.join(os.path.expanduser("~"), ".qaihm")
)
PY_PACKAGE_MODELS_ROOT = os.path.join(REPO_ROOT, PY_PACKAGE_RELATIVE_MODELS_ROOT)
STATIC_MODELS_ROOT = os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard", "static", "models")
SCORECARD_PACKAGE_MODELS_ROOT = os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard", "models")
SCORECARD_PACKAGE_MODELS_RELATIVE_ROOT = os.path.join(
    PY_PACKAGE_RELATIVE_SRC_ROOT, "scorecard", "models"
)

PUBLIC_BENCH_MODELS = os.path.join(
    PY_PACKAGE_SRC_ROOT, "scorecard", "static", "pytorch_bench_models_float.txt"
)

# Repo-relative dir the LLM compile/qdc tests write genie bundles to (must match GENIE_BUNDLES_ROOT in qai_hub_models.models.templates.llm.test).
GENIE_BUNDLES_ROOT = os.path.join(REPO_ROOT, "genie_bundles")

# Requirements Path
REQUIREMENTS_PATH = os.path.join(PY_PACKAGE_SRC_ROOT, "requirements.txt")
DEV_REQUIREMENTS_PATH = os.path.join(PY_PACKAGE_SRC_ROOT, "requirements-dev.txt")
GLOBAL_REQUIREMENTS_PATH = os.path.join(PY_PACKAGE_SRC_ROOT, "global_requirements.txt")
