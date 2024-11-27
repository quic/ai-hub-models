# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
from skimage import io

from qai_hub_models.models._shared.facemap_3dmm.app import FaceMap_3DMMApp
from qai_hub_models.models._shared.facemap_3dmm.demo import INPUT_IMAGE_PATH
from qai_hub_models.models.facemap_3dmm.test import GT_LMK_PATH
from qai_hub_models.models.facemap_3dmm_quantized.demo import main as demo_main
from qai_hub_models.models.facemap_3dmm_quantized.model import FaceMap_3DMMQuantizable


def test_task():
    app = FaceMap_3DMMApp(FaceMap_3DMMQuantizable.from_pretrained())
    input_image = io.imread(INPUT_IMAGE_PATH)

    # Get face bounding box info (from file or face detector)
    fbox = np.loadtxt(INPUT_IMAGE_PATH.replace(".jpg", "_fbox.txt"))
    x0, x1, y0, y1 = int(fbox[0]), int(fbox[1]), int(fbox[2]), int(fbox[3])

    lmk, output_image = app.landmark_prediction(input_image, x0, x1, y0, y1)
    lmk_gt = np.loadtxt(GT_LMK_PATH)

    assert (
        np.sqrt(np.sum((lmk.cpu().numpy() - lmk_gt) ** 2, 1).mean())
        / (x1 - x0 + 1)
        * 100
        < 3.0
    )


def test_demo():
    # Verify demo does not crash
    demo_main(is_test=True)
