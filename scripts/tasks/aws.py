# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Optional

from .constants import REPO_ROOT
from .task import ConditionalTask, NoOpTask, RunCommandsWithVenvTask
from .util import on_ci


class ValidateAwsCredentialsTask(ConditionalTask):
    def __init__(self, venv_path: Optional[str]):
        aws_script_path = f"{REPO_ROOT}/scripts/aws"
        install_awslogin_path = f"{aws_script_path}/install_awslogin.sh"
        validate_creds_path = f"{aws_script_path}/validate_credentials.py"
        super().__init__(
            group_name=None,
            condition=on_ci,
            true_task=NoOpTask("Not validating AWS credentials on CI"),
            false_task=RunCommandsWithVenvTask(
                "Validating AWS credentials",
                venv_path,
                [
                    f'if [ -d "{aws_script_path}" ]; then source {install_awslogin_path}; fi',
                    f'if [ -d "{aws_script_path}" ]; then python {validate_creds_path}; fi',
                ],
            ),
        )
