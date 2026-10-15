# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.templates.conformer.test import conformer_transcribe_test_e2e
from qai_hub_models.models.wav2vec2_conformer.app import Wav2Vec2ConformerApp
from qai_hub_models.models.wav2vec2_conformer.demo import load_demo_audio, main
from qai_hub_models.models.wav2vec2_conformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Wav2Vec2Conformer,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
)

GROUND_TRUTH_RESULT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ground_truth.txt"
)


def test_transcribe() -> None:
    model = Wav2Vec2Conformer.from_pretrained()
    app = Wav2Vec2ConformerApp(model)
    conformer_transcribe_test_e2e(app, load_demo_audio, GROUND_TRUTH_RESULT)


def test_demo() -> None:
    main(is_test=True)
