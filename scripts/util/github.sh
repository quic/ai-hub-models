#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"

GITHUB_ACTION=${GITHUB_ACTION:-}

on_ci() {
  if [ -n "${GITHUB_ACTION}" ]; then
    echo "1"
  fi
}

start_group() {
  group_name=$1

  if [ -n "$GITHUB_ACTION" ]; then
    echo "::group::$group_name"
  else
    echo -e "${COLOR_GREEN}$group_name${COLOR_OFF}"
  fi
}

end_group() {
  if [ -n "$GITHUB_ACTION" ]; then
    echo "::endgroup::"
  fi
}

set_github_output() {
  if [ -n "$GITHUB_ACTION" ]; then
    echo "$1=$2" >> "$GITHUB_OUTPUT"
  fi
}

warn() {
  message=$1

  if [ -n "$GITHUB_ACTION" ]; then
    echo "::warning::$message"
  else
    echo -e "${COLOR_RED}$message${COLOR_OFF}"
  fi
}
