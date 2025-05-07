# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
import functools
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .constants import (
    BASH_EXECUTABLE,
    PY_PACKAGE_MODELS_ROOT,
    process_output,
    run_and_get_output,
)


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[0;33m"
    OFF = "\033[0m"


@contextlib.contextmanager
def new_cd(x):
    d = os.getcwd()

    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)


@functools.cache
def check_code_gen_field(model_name: str, field_name: str) -> bool:
    """
    This process does not have the yaml package, so use this primitive way
    to check if a code gen field is true and apply branching logic within CI/scorecard.
    """
    yaml_path = Path(PY_PACKAGE_MODELS_ROOT) / model_name / "code-gen.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            if f"{field_name}: true" in f.read():
                return True
    return False


@functools.cache
def get_code_gen_str_field(model_name: str, field_name: str) -> str | None:
    """
    This process does not have the yaml package, so use this primitive way to get code-gen field value.
    """
    yaml_path = Path(PY_PACKAGE_MODELS_ROOT) / model_name / "code-gen.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            field = f"{field_name}:"
            for line in f.readlines():
                if line.startswith(field):
                    return line[len(field) : -1].strip("'").strip('"').strip()

    return None


def can_support_aimet(platform: str = sys.platform) -> bool:
    return (
        platform == "linux" or platform == "linux2"
    ) and sys.version_info.minor == 10


def get_is_hub_quantized(model_name) -> bool:
    return not check_code_gen_field(
        model_name, "is_precompiled"
    ) and not check_code_gen_field(model_name, "is_aimet")


def model_needs_aimet(model_name: str) -> bool:
    return check_code_gen_field(model_name, "is_aimet")


def get_model_python_version_requirements(
    model_name: str,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    # Returns minimum required version,
    # and "less than" required version (eg. python version must be less than the provided version)
    # None == No version set (any version OK)
    info = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "code-gen.yaml")
    req_less_than, min_version = None, None
    if os.path.exists(info):
        info_data = open(info).read()
        req_less_than = re.search(
            r'python_version_less_than:\s*["\']([\d.]+)["\']', info_data
        )
        if req_less_than:
            spl = req_less_than.group(1).split(".")
            req_less_than = int(spl[0]), int(spl[1])

        min_version = re.search(
            r'python_version_greater_than_or_equal_to:\s*["\']([\d.]+)["\']', info_data
        )
        if min_version:
            spl = min_version.group(1).split(".")
            min_version = int(spl[0]), int(spl[1])

    return (min_version, req_less_than)


def default_parallelism() -> int:
    """A conservative number of processes across which to spread pytests desiring parallelism."""
    from .github import on_github  # avoid circular import

    cpu_count = os.cpu_count()
    if not cpu_count:
        return 1

    # In CI, saturate the machine
    if on_github():
        return cpu_count

    # When running locally, leave a little CPU for other uses
    return max(1, int(cpu_count - 2))


# Convenience function for printing to stdout without buffering.
def echo(value, **args):
    print(value, flush=True, **args)


def have_root() -> bool:
    return os.geteuid() == 0


def on_linux():
    return platform.uname().system == "Linux"


def on_mac():
    return platform.uname().system == "Darwin"


def run(command):
    return subprocess.run(command, shell=True, check=True, executable=BASH_EXECUTABLE)


def run_with_venv(venv, command, env=None):
    if venv is not None:
        subprocess.run(
            f"source {venv}/bin/activate && {command}",
            shell=True,
            check=True,
            executable=BASH_EXECUTABLE,
            env=env,
        )
    else:
        run(command)


def run_with_venv_and_get_output(venv, command):
    if venv is not None:
        return process_output(
            subprocess.run(
                f"source {venv}/bin/activate && {command}",
                stdout=subprocess.PIPE,
                shell=True,
                check=True,
                executable=BASH_EXECUTABLE,
            )
        )
    else:
        return run_and_get_output(command)


def str_to_bool(word: str) -> bool:
    return word.lower() in ["1", "true", "yes"]


def get_env_bool(key: str, default: Optional[bool] = None) -> Optional[bool]:
    val = os.environ.get(key, None)
    if val is None:
        return None
    return str_to_bool(val)


def on_ci() -> bool:
    return get_env_bool("QAIHM_CI") or False


def debug_mode() -> bool:
    return get_env_bool("DEBUG_MODE") or False


@functools.cache
def uv_installed() -> bool:
    try:
        result = subprocess.run(["bash", "which", "uv"], capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


@functools.cache
def get_pip() -> str:
    # UV has trouble building many packages from source on Python 3.12
    if uv_installed() and (sys.version_info.major == 3 and sys.version_info.minor < 12):
        return "uv pip"
    else:
        return "pip"
