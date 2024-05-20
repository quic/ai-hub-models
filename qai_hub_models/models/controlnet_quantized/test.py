# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models.controlnet_quantized.demo import main as demo_main
from qai_hub_models.models.controlnet_quantized.export import export_model
from qai_hub_models.models.controlnet_quantized.model import ControlNetQuantized
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir


def test_from_precompiled():
    ControlNetQuantized.from_precompiled()


@pytest.mark.skip("#105 move slow_cloud and slow tests to nightly.")
@pytest.mark.slow_cloud
def test_export():
    with qaihm_temp_dir() as tmpdir:
        exported_jobs = export_model(
            # Testing text_encoder as it's smallest model in
            # ControlNet pipeline
            components=["text_encoder"],
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


@pytest.mark.skip("#105 move slow_cloud and slow tests to nightly.")
@pytest.mark.slow_cloud
def test_demo():
    demo_main(is_test=True)
