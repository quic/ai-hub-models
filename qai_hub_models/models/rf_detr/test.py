# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models._shared.detr.app import DETRApp
from qai_hub_models.models.rf_detr.demo import main as demo_main
from qai_hub_models.models.rf_detr.model import (
    DEFAULT_RESOLUTION,
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    RF_DETR,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

EXPECTED_OUTPUT = {75, 17}

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "detr_test_image.jpg"
)


def test_task() -> None:
    net = RF_DETR.from_pretrained()
    img = load_image(IMAGE_ADDRESS)
    _, _, label, _ = DETRApp(net, DEFAULT_RESOLUTION, DEFAULT_RESOLUTION).predict(
        img, DEFAULT_WEIGHTS
    )
    assert set(list(label.numpy())) == EXPECTED_OUTPUT


def test_cli_from_pretrained() -> None:
    args = get_model_cli_parser(RF_DETR).parse_args([])
    assert model_from_cli_args(RF_DETR, args) is not None


@pytest.mark.trace
def test_trace() -> None:
    net = RF_DETR.from_pretrained()
    input_spec = net.get_input_spec()
    trace = net.convert_to_torchscript(input_spec)

    img = load_image(IMAGE_ADDRESS)
    _, _, label, _ = DETRApp(trace, DEFAULT_RESOLUTION, DEFAULT_RESOLUTION).predict(
        img, DEFAULT_WEIGHTS
    )
    assert set(list(label.numpy())) == EXPECTED_OUTPUT


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
