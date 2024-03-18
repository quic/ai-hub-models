# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models._shared.detr.app import DETRApp
from qai_hub_models.models.detr_resnet101.demo import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.models.detr_resnet101.demo import main as demo_main
from qai_hub_models.models.detr_resnet101.model import DEFAULT_WEIGHTS, DETRResNet101
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "detr_test_image.jpg"
)


def test_task():
    net = DETRResNet101.from_pretrained(DEFAULT_WEIGHTS)
    img = load_image(IMAGE_ADDRESS)
    _, _, label, _ = DETRApp(net).predict(img, DEFAULT_WEIGHTS)
    assert set(list(label.numpy())) == {75, 63, 17}


@pytest.mark.trace
def test_trace():
    net = DETRResNet101.from_pretrained(DEFAULT_WEIGHTS).convert_to_torchscript()
    img = load_image(IMAGE_ADDRESS)
    _, _, label, _ = DETRApp(net).predict(img, DEFAULT_WEIGHTS)
    assert set(list(label.numpy())) == {75, 63, 17}


def test_demo():
    demo_main(is_test=True)
