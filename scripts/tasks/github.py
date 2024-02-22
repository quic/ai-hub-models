# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

from .util import Colors, echo


def on_github():
    return "GITHUB_ACTION" in os.environ


def start_group(group_name):
    if on_github():
        echo(f"::group::{group_name}")
    else:
        echo(f"{Colors.GREEN}{group_name}{Colors.OFF}")


def end_group():
    if on_github():
        echo("::endgroup::")


def set_github_output(key, value):
    if on_github():
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"{key}={value}", file=fh)
