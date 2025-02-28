# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models.huggingface_wavlm_base_plus.app import (
    HuggingFaceWavLMBasePlusApp,
)
from qai_hub_models.models.huggingface_wavlm_base_plus.demo import demo_main
from qai_hub_models.models.huggingface_wavlm_base_plus.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAMPLE_INPUTS,
    HuggingFaceWavLMBasePlus,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_TENSOR_1 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_tensor_1.npy"
)
OUTPUT_TENSOR_2 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_tensor_2.npy"
)


def _test_impl(app: HuggingFaceWavLMBasePlusApp) -> None:
    x = load_numpy(SAMPLE_INPUTS)["audio"]

    # Run inference
    app_output_features = app.predict_features(x)

    # Compare outputs
    np.testing.assert_allclose(
        np.asarray(app_output_features[0].detach().numpy(), dtype=np.float32),
        load_numpy(OUTPUT_TENSOR_1),
        rtol=0.02,
        atol=0.2,
    )

    np.testing.assert_allclose(
        np.asarray(app_output_features[1].detach().numpy(), dtype=np.float32),
        load_numpy(OUTPUT_TENSOR_2),
        rtol=0.02,
        atol=0.2,
    )


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
