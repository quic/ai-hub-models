# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models.baichuan2_7b import Model


@pytest.mark.skip(
    "https://github.com/qcom-ai-hub/tetracode/issues/13710 decide how to test pre-compiled LLMs."
)
@pytest.mark.slow_cloud
def test_load_model() -> None:
    Model.from_precompiled()
