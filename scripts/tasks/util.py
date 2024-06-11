# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import contextlib
import os
import platform
import subprocess
import sys


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


def can_support_aimet(platform: str = sys.platform) -> bool:
    return platform == "linux" or platform == "linux2"


def model_needs_aimet(model_name: str) -> bool:
    return "quantized" in model_name.lower()


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


def process_output(command):
    return command.stdout.decode("utf-8").strip()


def run(command):
    return subprocess.run(command, shell=True, check=True, executable=BASH_EXECUTABLE)


def run_and_get_output(command, check=True):
    return process_output(
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            shell=True,
            check=check,
            executable=BASH_EXECUTABLE,
        )
    )


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


BASH_EXECUTABLE = process_output(
    subprocess.run("which bash", stdout=subprocess.PIPE, shell=True, check=True)
)
