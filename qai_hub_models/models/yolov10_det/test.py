# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.yolov10_det.app import YoloV10DetectionApp
from qai_hub_models.models.yolov10_det.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov10_det.demo import main as demo_main
from qai_hub_models.models.yolov10_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV10Detector,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image.png"
)
WEIGHTS = "yolov10n.pt"


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = YoloV10DetectionApp(YoloV10Detector.from_pretrained(WEIGHTS))
    assert np.allclose(
        app.predict_boxes_from_image(image)[0], np.asarray(output_image), atol=1e3
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
