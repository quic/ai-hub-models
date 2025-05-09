# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"
. "${REPO_ROOT}/scripts/util/github.sh"
. "${REPO_ROOT}/scripts/ci/install-aws-cli.sh"

set_strict_mode

install_aws_cli
run_as_root apt-get install -y python3-opencv cmake libportaudio2

start_group "Ubuntu install: uv"
    if [ ! -x /usr/local/bin/uv ] || [ "$(uv --version)" != "uv 0.6.14" ]; then
        wget https://github.com/astral-sh/uv/releases/download/0.6.14/uv-installer.sh
        chmod +x uv-installer.sh
        run_as_root UV_UNMANAGED_INSTALL="/usr/local/bin" ./uv-installer.sh
        rm -rf uv-installer.sh
    else
        log_info "uv up to date."
    fi
end_group
