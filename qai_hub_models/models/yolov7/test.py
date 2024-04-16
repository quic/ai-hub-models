# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.models.yolov7.app import YoloV7DetectionApp
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov7.demo import main as demo_main
from qai_hub_models.models.yolov7.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV7,
    _load_yolov7_source_model_from_weights,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov7_demo_640_output.png"
)
WEIGHTS = "yolov7-tiny.pt"


@skip_clone_repo_check
def test_numerical():
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    source_model = _load_yolov7_source_model_from_weights(WEIGHTS)
    qaihm_model = YoloV7.from_pretrained(WEIGHTS)

    with torch.no_grad():
        # original model output
        source_model.model[-1].training = False
        source_model.model[-1].export = False
        source_detect_out = source_model(processed_sample_image)[0]
        source_out_postprocessed = detect_postprocess(source_detect_out)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(0, len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


@skip_clone_repo_check
def test_task():
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS).convert("RGB")
    app = YoloV7DetectionApp(YoloV7.from_pretrained(WEIGHTS))
    assert np.allclose(app.predict_boxes_from_image(image)[0], np.asarray(output_image))


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
