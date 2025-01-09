# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.video_classifier.app import KineticsClassifierApp
from qai_hub_models.models.resnet_mixed.demo import INPUT_VIDEO_PATH
from qai_hub_models.models.resnet_mixed.demo import main as demo_main
from qai_hub_models.models.resnet_mixed.model import ResNetMixed


def test_task():
    kinetics_app = KineticsClassifierApp(model=ResNetMixed.from_pretrained())
    dst_path = INPUT_VIDEO_PATH.fetch()
    prediction = kinetics_app.predict(path=dst_path)
    assert "surfing water" in prediction


def test_demo():
    demo_main(is_test=True)
