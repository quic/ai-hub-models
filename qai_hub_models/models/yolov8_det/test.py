# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch
from ultralytics import YOLO as ultralytics_YOLO

from qai_hub_models.models.yolov8_det.app import YoloV8DetectionApp
from qai_hub_models.models.yolov8_det.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov8_det.demo import main as demo_main
from qai_hub_models.models.yolov8_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV8Detector,
    yolov8_detect_postprocess,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image.png"
)
WEIGHTS = "yolov8n.pt"


@skip_clone_repo_check
def test_numerical():
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    source_model = ultralytics_YOLO(WEIGHTS).model
    qaihm_model = YoloV8Detector.from_pretrained(WEIGHTS)

    with torch.no_grad():
        # original model output
        source_detect_out, *_ = source_model(processed_sample_image)
        boxes, scores = torch.split(source_detect_out, [4, 80], 1)
        source_out_postprocessed = yolov8_detect_postprocess(boxes, scores)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(0, len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


@skip_clone_repo_check
def test_task():
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = YoloV8DetectionApp(YoloV8Detector.from_pretrained(WEIGHTS))
    assert np.allclose(app.predict_boxes_from_image(image)[0], np.asarray(output_image))


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
