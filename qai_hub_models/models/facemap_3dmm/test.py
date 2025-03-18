# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
from skimage import io

from qai_hub_models.models.facemap_3dmm.app import FaceMap_3DMMApp
from qai_hub_models.models.facemap_3dmm.demo import INPUT_FBOX_PATH, INPUT_IMAGE_PATH
from qai_hub_models.models.facemap_3dmm.demo import facemap_3dmm_demo as demo_main
from qai_hub_models.models.facemap_3dmm.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceMap_3DMM,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

GT_LMK_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "face_img_lmk.txt"
)


# Verify that the output from Torch is as expected.
def test_task() -> None:
    app = FaceMap_3DMMApp(FaceMap_3DMM.from_pretrained())
    input_image = io.imread(INPUT_IMAGE_PATH.fetch())

    # Get face bounding box info (from file or face detector)
    fbox = np.loadtxt(INPUT_FBOX_PATH.fetch())
    x0, x1, y0, y1 = (
        np.int32(fbox[0]),
        np.int32(fbox[1]),
        np.int32(fbox[2]),
        np.int32(fbox[3]),
    )

    lmk, output_image = app.landmark_prediction(input_image, x0, x1, y0, y1)
    lmk_gt = np.loadtxt(GT_LMK_PATH.fetch())

    assert (
        np.sqrt(np.sum((lmk.cpu().numpy() - lmk_gt) ** 2, 1).mean())
        / (x1 - x0 + 1)
        * 100
        < 1.0
    )


def test_trace() -> None:
    app = FaceMap_3DMMApp(FaceMap_3DMM.from_pretrained().convert_to_torchscript())
    input_image = io.imread(INPUT_IMAGE_PATH.fetch())

    # Get face bounding box info (from file or face detector)
    fbox = np.loadtxt(INPUT_FBOX_PATH.fetch())
    x0, x1, y0, y1 = (
        np.int32(fbox[0]),
        np.int32(fbox[1]),
        np.int32(fbox[2]),
        np.int32(fbox[3]),
    )

    lmk, _ = app.landmark_prediction(input_image, x0, x1, y0, y1)
    lmk_gt = np.loadtxt(GT_LMK_PATH.fetch())

    assert (
        np.sqrt(np.sum((lmk.cpu().numpy() - lmk_gt) ** 2, 1).mean())
        / (x1 - x0 + 1)
        * 100
        < 1.0
    )


def test_demo() -> None:
    demo_main(is_test=True)
