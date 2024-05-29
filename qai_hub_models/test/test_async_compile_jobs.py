# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import datetime
import os

import qai_hub as hub

from qai_hub_models.utils.asset_loaders import load_yaml


def test_compile_jobs_success():
    """
    When testing compilation in CI, synchronously waiting for each compile_job to
    finish is too slow. Instead, job ids are written to a file upon submission,
    and success is validated all at once in the end using this test.
    """
    if os.stat(os.environ["COMPILE_JOBS_FILE"]).st_size == 0:
        return
    job_ids = load_yaml(os.environ["COMPILE_JOBS_FILE"])

    failed_jobs = {}
    timeout_jobs = {}
    for name, job_id in job_ids.items():
        job = hub.get_job(job_id)
        if job.get_status().running:
            # Wait a maximum of 15 minutes for a compile job
            timemax = datetime.timedelta(minutes=15)
            timediff = datetime.datetime.now() - job.date
            if timediff < timemax:
                try:
                    job = job.wait((timemax - timediff).total_seconds())
                except TimeoutError:
                    timeout_jobs[name] = job_id
            else:
                timeout_jobs[name] = job_id
        elif not job.get_status().success:
            failed_jobs[name] = job_id

    error_strs = []
    if failed_jobs:
        error_strs.append(f"The following jobs failed to compile: {failed_jobs}")
    if timeout_jobs:
        error_strs.append(f"The following jobs timed out: {timeout_jobs}")
    if len(error_strs) > 0:
        raise ValueError("\n".join(error_strs))
