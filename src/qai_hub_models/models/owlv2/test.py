# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.owlv2.app import OwlV2CollectionApp
from qai_hub_models.models.owlv2.demo import DEFAULT_TEXT_QUERIES, IMAGE_ADDRESS
from qai_hub_models.models.owlv2.demo import main as demo_main
from qai_hub_models.models.owlv2.model import (
    HF_MODEL_ID,
    IMAGE_SIZE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OwlV2,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.test_helpers import assert_most_close

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/owlv2_output_image.png"
)


def test_task() -> None:
    collection_model = OwlV2.from_pretrained()
    vision_encoder = collection_model.vision_encoder
    text_detector = collection_model.text_detector
    app = OwlV2CollectionApp(
        vision_encoder, text_detector, HF_MODEL_ID, IMAGE_SIZE, IMAGE_SIZE
    )
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
