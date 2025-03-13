# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models.nomic_embed_text.app import NomicEmbedTextApp
from qai_hub_models.models.nomic_embed_text.demo import main as demo_main
from qai_hub_models.models.nomic_embed_text.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NomicEmbedText,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.testing import skip_clone_repo_check

EMBEDDINGS_GT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "embeddings.npy"
)

SAMPLE_TEXT = "Hello world!"


@skip_clone_repo_check
def test_task():
    model = NomicEmbedText.from_pretrained()
    app = NomicEmbedTextApp(model, model.seq_length)
    embeddings = app.predict(SAMPLE_TEXT)
    np.testing.assert_allclose(
        embeddings, load_numpy(EMBEDDINGS_GT), rtol=1e-5, atol=0.01
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    model = NomicEmbedText.from_pretrained()
    app = NomicEmbedTextApp(model.convert_to_torchscript(), model.seq_length)
    embeddings = app.predict(SAMPLE_TEXT)
    np.testing.assert_allclose(
        embeddings, load_numpy(EMBEDDINGS_GT), rtol=1e-5, atol=0.01
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True, default_text=SAMPLE_TEXT)
