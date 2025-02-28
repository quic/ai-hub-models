#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# shellcheck source=/dev/null

REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"

set_strict_mode

cd "$(dirname "$0")/../.."

# Uncomment "echo"s below to debug

venv="${VENV_PATH:-qaihm-dev}"
# echo "Activating venv in ${venv}"
source "${venv}/bin/activate"

if [ "$#" -eq 0 ]; then
  mypy_files=(qai_hub_models)
else
  mypy_files=("$@")
fi

# echo "Checking ${mypy_files[*]}"
mypy --ignore-missing-imports --warn-unused-configs --config-file="${REPO_ROOT}/mypy.ini" "${mypy_files[@]}"
