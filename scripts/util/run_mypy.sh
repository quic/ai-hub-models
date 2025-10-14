#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
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

if [[ "$#" -eq 0 || "$#" -gt 100 ]]; then
  # If the changed file list is empty or especially large, just check the whole package.
  mypy_files=("-p" "qai_hub_models")
else
  mypy_files=("$@")
fi

# echo "Checking ${mypy_files[*]}"
mypy --warn-unused-configs --config-file="${REPO_ROOT}/pyproject.toml" "${mypy_files[@]}"
