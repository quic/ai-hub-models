# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.easyocr.app import EasyOCRApp
from qai_hub_models.models.easyocr.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.easyocr.demo import main as demo_main
from qai_hub_models.models.easyocr.model import (
    LANG_LIST,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EasyOCR,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

FIRST_LINE_IMAGE_TEXT = "Reduce your risk of coronavirus infection:"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "english_output_with_bbox.png"
)
CHINESE_INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "chinese.jpg"
)


@skip_clone_repo_check
def test_task():
    model = EasyOCR.from_pretrained(LANG_LIST)
    app = EasyOCRApp(model.detector, model.recognizer, model.lang_list)
    image = load_image(INPUT_IMAGE_ADDRESS)
    results = app.predict_text_from_image(image)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)

    # Checking bounding box
    assert np.allclose(np.asarray(results[0]), np.asarray(output_image))

    # Checking recognition text
    assert results[1][0][1] == FIRST_LINE_IMAGE_TEXT

    ch_model = EasyOCR.from_pretrained(["ch_sim", "en"])
    app = EasyOCRApp(ch_model.detector, ch_model.recognizer, ch_model.lang_list)
    image = load_image(CHINESE_INPUT_IMAGE_ADDRESS)
    results = app.predict_text_from_image(image)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)

    # Checking recognition text
    assert results[1][0][1] == "西"


def test_demo():
    demo_main(is_test=True)
