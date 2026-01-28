# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

from .util import Colors, echo


def on_github() -> bool:
    return "GITHUB_ACTION" in os.environ


def start_group(group_name: str) -> None:
    if on_github():
        echo(f"::group::{group_name}")
    else:
        echo(f"{Colors.GREEN}{group_name}{Colors.OFF}")


def end_group() -> None:
    if on_github():
        echo("::endgroup::")


def set_github_output(key: str, value: str) -> None:
    if on_github():
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"{key}={value}", file=fh)


def append_to_summary(content: str) -> None:
    """Append content to GitHub step summary."""
    if on_github() and "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            print(content, file=fh)
    else:
        # Print to console when not on GitHub
        echo(content)
