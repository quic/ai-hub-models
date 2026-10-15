# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.owl_vit.demo import DEFAULT_TEXT_QUERIES, IMAGE_ADDRESS
from qai_hub_models.models.owl_vit.demo import main as demo_main
from qai_hub_models.models.owl_vit.model import (
    HF_MODEL_ID,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OwlViT,
)
from qai_hub_models.models.templates.owl.app import OwlApp
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.test_helpers import assert_most_close

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image.png"
)


def test_task() -> None:
    app = OwlApp(OwlViT.from_pretrained(), HF_MODEL_ID)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    image = load_image(IMAGE_ADDRESS)
    assert_most_close(
        np.asarray(app.predict_boxes_from_image(image, DEFAULT_TEXT_QUERIES)[0]),
        np.asarray(output_image),
        0.05,
        0.05,
        1,
    )


def test_demo() -> None:
    demo_main(is_test=True)
