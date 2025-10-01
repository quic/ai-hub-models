# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.utils.envvar_bases import (
    QAIHMBoolEnvvar,
)


class DisableCompileJobTimeoutEnvvar(QAIHMBoolEnvvar):
    """
    If this is false, a 1 hr timeout is enforced on compile jobs, post submission time.
    This is a separate timeout than the timeout on AI Hub. This timeout is intended for
    PR tests, so users don't have to wait hours for hub to time out their job to know it's failing.

    If this is true, the override timeout is disabled for AI Hub Models testing.
    We will wait for the job to time out on AI Hub instead."
    """

    VARNAME = "QAIHM_TEST_DISABLE_COMPILE_JOB_TIMEOUT"
    CLI_ARGNAMES = ["--disable-cj-timeout"]
    CLI_HELP_MESSAGE = "For testing, AI Hub Models enforces a 1 hour timeout on compile jobs (after submission time) by default. If True, the QAIHM-specific 1hr timeout is disabled."

    @classmethod
    def default(cls):
        return False


__all__ = ["DisableCompileJobTimeoutEnvvar"]
