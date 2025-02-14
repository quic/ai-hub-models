# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torch
from ultralytics import YOLO as ultralytics_YOLO

from qai_hub_models.models._shared.yolo.app import YoloSegmentationApp
from qai_hub_models.models._shared.yolo.model import yolo_segment_postprocess
from qai_hub_models.models.yolov11_seg.demo import IMAGE_ADDRESS, OUTPUT_IMAGE_ADDRESS
from qai_hub_models.models.yolov11_seg.demo import main as demo_main
from qai_hub_models.models.yolov11_seg.model import NUM_ClASSES, YoloV11Segmentor
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

WEIGHTS = "yolo11n-seg.pt"


@skip_clone_repo_check
def test_task() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    qaihm_model = YoloV11Segmentor.from_pretrained(WEIGHTS)
    qaihm_app = YoloSegmentationApp(qaihm_model)
    source_model = ultralytics_YOLO(WEIGHTS).model

    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    processed_sample_image = qaihm_app.preprocess_input(processed_sample_image)

    with torch.no_grad():
        # original model output
        source_out = source_model(processed_sample_image)
        source_out_postprocessed = yolo_segment_postprocess(source_out[0], NUM_ClASSES)
        source_out = [*source_out_postprocessed, source_out[1][-1]]

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(0, len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


@skip_clone_repo_check
@pytest.mark.trace
def test_trace() -> None:
    net = YoloV11Segmentor.from_pretrained(WEIGHTS)
    input_spec = net.get_input_spec()
    trace = net.convert_to_torchscript(input_spec, check_trace=False)

    # Collect output via app for traced model
    img = load_image(IMAGE_ADDRESS)
    app = YoloSegmentationApp(trace)
    out_imgs = app.predict(img)

    expected_out = load_image(OUTPUT_IMAGE_ADDRESS)
    assert_most_close(
        np.asarray(out_imgs[0], dtype=np.float32),
        np.asarray(expected_out, dtype=np.float32),
        0.005,
        rtol=0.02,
        atol=1.5,
    )


@skip_clone_repo_check
def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
