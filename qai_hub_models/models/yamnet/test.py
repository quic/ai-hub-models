# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.yamnet.app import YamNetApp
from qai_hub_models.models.yamnet.demo import INPUT_AUDIO_ADDRESS
from qai_hub_models.models.yamnet.demo import main as demo_main
from qai_hub_models.models.yamnet.model import YamNet
from qai_hub_models.utils.testing import skip_clone_repo_check


# Verify that the output from Torch is as expected.
@skip_clone_repo_check
def test_task():
    yamnet_app = YamNetApp(model=YamNet.from_pretrained())
    dst_path = INPUT_AUDIO_ADDRESS.fetch()
    prediction = yamnet_app.predict(path=dst_path)
    assert "Whistling" in prediction


def test_demo():
    demo_main(is_test=True)
