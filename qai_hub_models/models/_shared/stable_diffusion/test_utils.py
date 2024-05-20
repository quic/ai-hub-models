# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir


def export_for_component(export_model, component_name: str):
    with qaihm_temp_dir() as tmpdir:
        exported_jobs = export_model(
            # Testing text_encoder as it's smallest model in
            # Stable-Diffusion pipeline
            components=[component_name],
            skip_inferencing=True,
            skip_downloading=True,
            skip_summary=True,
            output_dir=tmpdir,
        )

        # NOTE: Not waiting for job to finish
        # as it will slow CI down.
        # Rather, we should create waiting test and move to nightly.
        for jobs in exported_jobs.values():
            profile_job, inference_job = jobs[0], jobs[1]
            assert profile_job is not None
            assert inference_job is None
