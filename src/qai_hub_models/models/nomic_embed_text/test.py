# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.nomic_embed_text.app import NomicEmbedTextApp
from qai_hub_models.models.nomic_embed_text.demo import main as demo_main
from qai_hub_models.models.nomic_embed_text.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NomicEmbedText,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy

EMBEDDINGS_GT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "embeddings.npy"
)

SAMPLE_TEXT = "Hello world!"


def test_task() -> None:
    model = NomicEmbedText.from_pretrained()
    app = NomicEmbedTextApp(model, model.seq_length)
    embeddings = app.predict(SAMPLE_TEXT)
    np.testing.assert_allclose(
        embeddings, load_numpy(EMBEDDINGS_GT), rtol=1e-5, atol=0.01
    )


def test_demo() -> None:
    demo_main(is_test=True, default_text=SAMPLE_TEXT)
