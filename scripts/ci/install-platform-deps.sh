# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"
. "${REPO_ROOT}/scripts/ci/install-aws-cli.sh"

set_strict_mode

install_aws_cli
run_as_root apt-get install -y python3-opencv
