#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# This should be sourced and hence does not specify an interpreter.

REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"

set_strict_mode

# Path to the virtual environment, relative to the repository root.
ENV_PATH="qaihm-dev"

# command flag options
# Parse command line configure flags ------------------------------------------
while [ $# -gt 0 ]
  do case $1 in
    --venv=*)            ENV_PATH=${1##--venv=} ;;
    *) echo "Bad opt $1." && exit 1;;
  esac
  shift
done

python3 "${REPO_ROOT}/scripts/build_and_test.py" --venv="${ENV_PATH}" install_deps
