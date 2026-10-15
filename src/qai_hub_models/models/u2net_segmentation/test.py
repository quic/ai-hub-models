# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from PIL import Image

from qai_hub_models.models.u2net_segmentation.app import SegmentationU2NetApp
from qai_hub_models.models.u2net_segmentation.demo import main as demo_main
from qai_hub_models.models.u2net_segmentation.model import (
    IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SegmentationU2Net,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "u2net_test_output_new.png"
)


def test_task() -> None:
    """Test PyTorch model produces correct output matching reference."""
    model = SegmentationU2Net.from_pretrained()
    app = SegmentationU2NetApp(model, model.get_input_spec())

    image = load_image(IMAGE_ADDRESS).convert("RGB")
    output = app.predict(image)

    assert isinstance(output, Image.Image)
    assert output.size == image.size

    expected_out = load_image(OUTPUT_ADDRESS)
    np.testing.assert_allclose(np.array(output), np.array(expected_out), atol=1)


def test_demo() -> None:
    """Test that demo runs without exceptions."""
    demo_main(is_test=True)
