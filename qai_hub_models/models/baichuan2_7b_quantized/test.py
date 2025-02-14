# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models.baichuan2_7b_quantized import Model


@pytest.mark.skip(
    "https://github.com/qcom-ai-hub/tetracode/issues/13710 decide how to test pre-compiled LLMs."
)
@pytest.mark.slow_cloud
def test_load_model() -> None:
    Model.from_precompiled()
