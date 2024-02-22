# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

import qai_hub as hub
import yaml


def test_compile_jobs_success():
    """
    When testing compilation in CI, synchronously waiting for each compile_job to
    finish is too slow. Instead, job ids are written to a file upon submission,
    and success is validated all at once in the end using this test.
    """
    if os.stat(os.environ["COMPILE_JOBS_FILE"]).st_size == 0:
        return
    with open(os.environ["COMPILE_JOBS_FILE"], "r") as f:
        job_ids = yaml.safe_load(f.read())
    failed_jobs = {}
    for name, job_id in job_ids.items():
        result = hub.get_job(job_id).wait()
        if not result.success:
            failed_jobs[name] = job_id
    if failed_jobs:
        raise ValueError(f"The following jobs failed to compile: {failed_jobs}")
