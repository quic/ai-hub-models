# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.models.yolov5.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov5.demo import main as demo_main
from qai_hub_models.models.yolov5.model import (
    DEFAULT_WEIGHTS,
    YoloV5,
    _load_yolov5_source_model_from_weights,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.testing import skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    # source model
    source_model = _load_yolov5_source_model_from_weights(DEFAULT_WEIGHTS)

    # Qualcomm AI Hub Model
    qaihm_model = YoloV5.from_pretrained()

    with torch.no_grad():
        # source model output
        processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
        source_detect_out = source_model(processed_sample_image)
        source_out_postprocessed = detect_postprocess(
            source_detect_out[0][0][None, :, :]
        )

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(0, len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
