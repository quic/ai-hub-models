# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
from PIL import Image
from ultralytics.models.fastsam import FastSAM, FastSAMPrompt

from qai_hub_models.models._shared.fastsam.app import FastSAMApp
from qai_hub_models.models.fastsam_x.demo import INPUT_IMAGE
from qai_hub_models.models.fastsam_x.demo import main as demo_main
from qai_hub_models.models.fastsam_x.model import DEFAULT_WEIGHTS, FastSAM_X
from qai_hub_models.utils.image_processing import preprocess_PIL_image


def test_task() -> None:
    image_path = INPUT_IMAGE.fetch()
    image = Image.open(image_path)
    image = preprocess_PIL_image(image)
    app = FastSAMApp(FastSAM_X.from_pretrained())  # type: ignore[call-arg]
    result, _ = app.segment_image(str(image_path))

    model = FastSAM(DEFAULT_WEIGHTS)
    everything_results = model(
        image_path, device="cpu", retina_masks=True, imgsz=640, conf=0.4, iou=0.9
    )
    prompt = FastSAMPrompt(image_path, everything_results, device="cpu")
    predictions = prompt.everything_prompt()

    assert result[0].masks is not None
    assert np.allclose(result[0].masks.data, predictions[0].masks.data)


def test_demo() -> None:
    demo_main(is_test=True)
