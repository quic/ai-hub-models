# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models.litehrnet.app import LiteHRNetApp
from qai_hub_models.models.litehrnet.demo import IMAGE_ADDRESS
from qai_hub_models.models.litehrnet.demo import main as demo_main
from qai_hub_models.models.litehrnet.model import LiteHRNet
from qai_hub_models.utils.asset_loaders import load_image

EXPECTED_KEYPOINTS = np.array(
    [
        [
            [70, 34],
            [77, 32],
            [72, 30],
            [91, 37],
            [72, 32],
            [109, 67],
            [67, 67],
            [130, 104],
            [63, 104],
            [112, 125],
            [40, 102],
            [105, 144],
            [77, 144],
            [119, 202],
            [81, 190],
            [142, 251],
            [88, 230],
        ]
    ]
)


def _test_impl(app: LiteHRNetApp):
    image = load_image(IMAGE_ADDRESS)
    keypoints = app.predict_pose_keypoints(image, True)

    np.testing.assert_allclose(
        np.asarray(EXPECTED_KEYPOINTS, dtype=np.float32),
        np.asarray(keypoints, dtype=np.float32),
        rtol=0.02,
        atol=1.5,
    )


def test_task():
    litehrnet = LiteHRNet.from_pretrained()
    _test_impl(LiteHRNetApp(litehrnet, litehrnet.inferencer))


@pytest.mark.trace
def test_trace():
    litehrnet = LiteHRNet.from_pretrained()
    _test_impl(LiteHRNetApp(litehrnet.convert_to_torchscript(), litehrnet.inferencer))


def test_demo():
    demo_main(is_test=True)
