# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"

function install_aws_cli() {
    # Install AWS CLI
    AWS_VERSION=2.22.0

    if ! command -v aws &> /dev/null || [[ "$(aws --version)" != *"aws-cli/${AWS_VERSION}"* ]]
    then
      curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWS_VERSION}.zip" -o "/tmp/awscliv2.zip"
      unzip /tmp/awscliv2.zip -d /tmp
      run_as_root /tmp/aws/install --update
      rm -rf /tmp/aws /tmp/awscliv2.zip

      echo "$(aws --version)"
    fi
}
