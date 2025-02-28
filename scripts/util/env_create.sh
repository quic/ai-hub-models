#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# shellcheck source=/dev/null # we are statically sourcing a script.
# This can be sourced and hence does not specify an interpreter.

orig_flags=$-

set -e

# Path to the virtual environment, relative to the repository root.
ENV_PATH="qaihm-dev"

SYNC=1

PYTHON="python3.10"

# command flag options
# Parse command line configure flags ------------------------------------------
while [ $# -gt 0 ]
  do case $1 in
    --venv=*)            ENV_PATH=${1##--venv=} ;;
    --no-sync)           SYNC=0 ;;
    --python=*)          PYTHON=${1##--python=} ;;
    *) echo "Bad opt $1." && exit 1;;
  esac
  shift
done

if [ ! -d "$ENV_PATH" ]; then
  mkdir -p "$(dirname "$ENV_PATH")"

  echo "Creating virtual env $ENV_PATH."
  $PYTHON -m venv "$ENV_PATH"

  echo "Activating virtual env."
  source "$ENV_PATH/bin/activate"
  # setuptools also needed for build from source
  pip install pip==24.3.1 setuptools
else
  source "$ENV_PATH/bin/activate"
  echo "Env created already. Skipping creation."
fi

if [ $SYNC -eq 1 ]; then
    source scripts/util/env_sync.sh --venv="$ENV_PATH"
fi

# Unset -e so our shell doesn't close the next time something exits with
# non-zero status.
if [[ ! "${orig_flags}" =~ e ]]; then
  set +e
fi
