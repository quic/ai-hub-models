# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torch
from datasets import load_dataset

from qai_hub_models.models.huggingface_wavlm_base_plus.app import (
    HuggingFaceWavLMBasePlusApp,
)
from qai_hub_models.models.huggingface_wavlm_base_plus.demo import demo_main
from qai_hub_models.models.huggingface_wavlm_base_plus.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    HuggingFaceWavLMBasePlus,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_TENSOR_1 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "wavlm_output_tensor_1.pth"
)
OUTPUT_TENSOR_2 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "wavlm_output_tensor_2.pth"
)


def _test_impl(app: HuggingFaceWavLMBasePlusApp):
    # Load input data
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
    )
    dataset = dataset.sort("id")
    x = dataset[0]["audio"]["array"]
    sampling_rate = dataset.features["audio"].sampling_rate

    # Load expected output data
    first_output_tensor = torch.load(OUTPUT_TENSOR_1.fetch())
    output_array1 = first_output_tensor.detach().numpy()
    second_output_tensor = torch.load(OUTPUT_TENSOR_2.fetch())
    output_array2 = second_output_tensor.detach().numpy()

    # Run inference
    app_output_features = app.predict_features(x, sampling_rate)

    # Compare outputs
    np.testing.assert_allclose(
        np.asarray(app_output_features[0].detach().numpy(), dtype=np.float32),
        np.asarray(output_array1, dtype=np.float32),
        rtol=0.02,
        atol=0.2,
    )

    np.testing.assert_allclose(
        np.asarray(app_output_features[1].detach().numpy(), dtype=np.float32),
        np.asarray(output_array2, dtype=np.float32),
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_task():
    _test_impl(HuggingFaceWavLMBasePlusApp(HuggingFaceWavLMBasePlus.from_pretrained()))


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    _test_impl(
        HuggingFaceWavLMBasePlusApp(
            HuggingFaceWavLMBasePlus.from_pretrained().convert_to_torchscript()
        )
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
