# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.models._shared.yolo.model import yolo_detect_postprocess
from qai_hub_models.models.yolov11_det.app import YoloV11DetectionApp
from qai_hub_models.models.yolov11_det.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov11_det.demo import main as demo_main
from qai_hub_models.models.yolov11_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV11Detector,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    wipe_sys_modules,
)
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image.png"
)
WEIGHTS = "yolo11n.pt"


@skip_clone_repo_check
def test_numerical() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    import ultralytics

    wipe_sys_modules(ultralytics)
    from ultralytics import YOLO as ultralytics_YOLO

    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    qaihm_model = YoloV11Detector.from_pretrained(WEIGHTS)
    source_model = ultralytics_YOLO(WEIGHTS).model

    with torch.no_grad():
        # original model output
        (boxes, scores), _ = source_model(processed_sample_image)

        # boxes, scores = torch.split(source_detect_out, [4, 8400], 1)
        source_out_postprocessed = yolo_detect_postprocess(boxes, scores)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(0, len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = YoloV11DetectionApp(YoloV11Detector.from_pretrained(WEIGHTS))
    assert np.allclose(app.predict_boxes_from_image(image)[0], np.asarray(output_image))


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
