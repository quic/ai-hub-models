# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torch

from qai_hub_models.models.stylegan2.app import StyleGAN2App
from qai_hub_models.models.stylegan2.demo import main as demo_main
from qai_hub_models.models.stylegan2.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    StyleGAN2,
    _load_stylegan2_source_model_from_weights,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

SAMPLE_GENERATOR_RANDOM_SEED = 1000
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_sample_image.png"
)


@skip_clone_repo_check
def test_task():
    source_model = _load_stylegan2_source_model_from_weights(DEFAULT_WEIGHTS)
    qaihm_model = StyleGAN2.from_pretrained(DEFAULT_WEIGHTS)
    z = StyleGAN2App(qaihm_model).generate_random_vec(seed=SAMPLE_GENERATOR_RANDOM_SEED)

    with torch.no_grad():
        assert_most_close(
            source_model(z, [[]], noise_mode="const", force_fp32=True),
            qaihm_model(z),
            0.005,
        )


@skip_clone_repo_check
def test_stylegan2_app():
    app = StyleGAN2App(StyleGAN2.from_pretrained())

    # App generates expected image
    z = app.generate_random_vec(seed=SAMPLE_GENERATOR_RANDOM_SEED)
    expected = np.asarray(load_image(OUTPUT_IMAGE_ADDRESS).convert("RGB"))
    output = np.asarray(app.generate_images(z, raw_output=True))
    assert_most_close(output, expected, 0.005)

    # App can generate multiple images
    output_images = app.generate_images(class_idx=torch.Tensor([1, 2]).type(torch.int))
    assert len(output_images) == 2


@pytest.mark.trace
@skip_clone_repo_check
def test_stylegan2_trace():
    app = StyleGAN2App(StyleGAN2.from_pretrained().convert_to_torchscript())

    # App generates expected image
    z = app.generate_random_vec(seed=SAMPLE_GENERATOR_RANDOM_SEED)
    expected = np.asarray(load_image(OUTPUT_IMAGE_ADDRESS).convert("RGB"))
    output = np.asarray(app.generate_images(z, raw_output=True))[0]

    assert_most_close(output, expected, 0.005)


@skip_clone_repo_check
def test_stylegan2_demo():
    # Verify demo does not crash
    demo_main(is_test=True)
