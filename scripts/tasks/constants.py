# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

from .util import run_and_get_output

# Env Variable
STORE_ROOT_ENV_VAR = "QAIHM_STORE_ROOT"

# Repository
REPO_ROOT = run_and_get_output("git rev-parse --show-toplevel")
VENV_PATH = os.path.join(REPO_ROOT, "qaihm-dev")
BUILD_ROOT = os.path.join(REPO_ROOT, "build")

# Dependent Wheels
QAI_HUB_LATEST_PATH = os.path.join(BUILD_ROOT, "qai_hub-latest-py3-none-any.whl")

# Package paths relative to repository root
PY_PACKAGE_RELATIVE_SRC_ROOT = "qai_hub_models"
PY_PACKAGE_RELATIVE_MODELS_ROOT = os.path.join(PY_PACKAGE_RELATIVE_SRC_ROOT, "models")

# Absolute package paths
PY_PACKAGE_INSTALL_ROOT = REPO_ROOT
PY_PACKAGE_SRC_ROOT = os.path.join(
    PY_PACKAGE_INSTALL_ROOT, PY_PACKAGE_RELATIVE_SRC_ROOT
)
PY_PACKAGE_LOCAL_CACHE = os.environ.get(
    STORE_ROOT_ENV_VAR, os.path.join(os.path.expanduser("~"), ".qaihm")
)
PY_PACKAGE_MODELS_ROOT = os.path.join(
    PY_PACKAGE_INSTALL_ROOT, PY_PACKAGE_RELATIVE_MODELS_ROOT
)
