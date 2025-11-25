# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.utils.envvar_bases import QAIHMBoolEnvvar


class IsOnCIEnvvar(QAIHMBoolEnvvar):
    """If this is true, the tests are running in a continuous integration environment."""

    VARNAME = "QAIHM_CI"
    CLI_ARGNAMES = ["--ci-mode"]
    CLI_HELP_MESSAGE = "If set, acts as if the script is running in CI."

    @classmethod
    def default(cls):
        return False
