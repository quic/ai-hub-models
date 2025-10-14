# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models.huggingface_wavlm_base_plus.app import (
    HuggingFaceWavLMBasePlusApp,
)
from qai_hub_models.models.huggingface_wavlm_base_plus.demo import demo_main
from qai_hub_models.models.huggingface_wavlm_base_plus.model import (
    SAMPLE_INPUTS,
    HuggingFaceWavLMBasePlus,
)
from qai_hub_models.utils.asset_loaders import load_numpy
from qai_hub_models.utils.testing import skip_clone_repo_check


def _test_impl(app: HuggingFaceWavLMBasePlusApp) -> None:
    x = load_numpy(SAMPLE_INPUTS)["audio"]

    app_output = app.predict_features(x)
    expected_text = "and so my fellow americas ask not what your country can do for you ask what you can do for your"
    assert app_output == expected_text


@skip_clone_repo_check
def test_task() -> None:
    _test_impl(HuggingFaceWavLMBasePlusApp(HuggingFaceWavLMBasePlus.from_pretrained()))


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    _test_impl(
        HuggingFaceWavLMBasePlusApp(
            HuggingFaceWavLMBasePlus.from_pretrained().convert_to_torchscript()
        )
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
