#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# A shell script that prints the value of the SUDO_PASSWORD environment variable to stdout.
# The sole purpose of this script is so that we can use it with sudo's --askpass argument in CI.

if [ -z "${SUDO_PASSWORD}" ]; then
    echo "SUDO_PASSWORD is not set" >&2
    exit 1
else
    echo "${SUDO_PASSWORD}"
fi
