# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.funasr_conformer_en.app import FunASRConformerEnApp
from qai_hub_models.models.funasr_conformer_en.demo import load_demo_audio, main
from qai_hub_models.models.funasr_conformer_en.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FunASRConformerEn,
)
from qai_hub_models.models.templates.conformer.test import conformer_transcribe_test_e2e
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

GROUND_TRUTH_RESULT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ground_truth.txt"
)


def test_transcribe() -> None:
    model = FunASRConformerEn.from_pretrained()
    app = FunASRConformerEnApp(model, model.frontend, model.token_list)
    conformer_transcribe_test_e2e(app, load_demo_audio, GROUND_TRUTH_RESULT)


def test_demo() -> None:
    main(is_test=True)
