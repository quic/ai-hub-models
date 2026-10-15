# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import numpy as np

from qai_hub_models.models.act.app import ACTApp
from qai_hub_models.models.act.demo import main as demo_main
from qai_hub_models.models.act.model import (
    ACT,
    MODEL_ASSET_VERSION,
    MODEL_ID,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_numpy,
)

OUTPUT_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_actions.npy"
)


# Verify that the output from Torch is as expected.
def test_task() -> None:
    app = ACTApp(ACT.from_pretrained())
    actions = app.predict(episode_len=1, raw_output=True)

    output = load_numpy(OUTPUT_ADDRESS)

    np.testing.assert_allclose(
        np.asarray(actions), np.asarray(output), atol=0.8, rtol=0.1
    )


def test_demo() -> None:
    demo_main(is_test=True)
