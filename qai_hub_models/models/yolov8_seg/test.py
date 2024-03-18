# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torch
from ultralytics import YOLO as ultralytics_YOLO

from qai_hub_models.models.yolov8_seg.app import YoloV8SegmentationApp
from qai_hub_models.models.yolov8_seg.demo import IMAGE_ADDRESS, OUTPUT_IMAGE_ADDRESS
from qai_hub_models.models.yolov8_seg.demo import main as demo_main
from qai_hub_models.models.yolov8_seg.model import (
    YoloV8Segmentor,
    yolov8_segment_postprocess,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.testing import assert_most_close

WEIGHTS = "yolov8n-seg.pt"


def test_task():
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    source_model = ultralytics_YOLO(WEIGHTS).model
    qaihm_model = YoloV8Segmentor.from_pretrained(WEIGHTS)
    qaihm_app = YoloV8SegmentationApp(qaihm_model)
    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    processed_sample_image = qaihm_app.preprocess_input(processed_sample_image)

    with torch.no_grad():
        # original model output
        source_out = source_model(processed_sample_image)
        source_out_postprocessed = yolov8_segment_postprocess(source_out[0])
        source_out = [*source_out_postprocessed, source_out[1][-1]]

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(0, len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


@pytest.mark.trace
def test_trace():
    net = YoloV8Segmentor.from_pretrained(WEIGHTS)
    input_spec = net.get_input_spec()
    trace = net.convert_to_torchscript(input_spec, check_trace=False)

    # Collect output via app for traced model
    img = load_image(IMAGE_ADDRESS)
    app = YoloV8SegmentationApp(trace)
    out_imgs = app.predict(img)

    expected_out = load_image(OUTPUT_IMAGE_ADDRESS)
    assert_most_close(
        np.asarray(out_imgs[0], dtype=np.float32),
        np.asarray(expected_out, dtype=np.float32),
        0.005,
        rtol=0.02,
        atol=1.5,
    )


def test_demo():
    # Run demo and verify it does not crash
    demo_main(is_test=True)
