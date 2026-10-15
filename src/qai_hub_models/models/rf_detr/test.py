# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models.rf_detr.demo import main as demo_main
from qai_hub_models.models.rf_detr.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    RF_DETR,
    SUPPORTED_VARIANTS,
    VARIANT_RESOLUTION,
)
from qai_hub_models.models.templates.detr.app import DETRApp
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

EXPECTED_OUTPUT = {75, 17}

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "detr_test_image.jpg"
)


def test_task() -> None:
    net = RF_DETR.from_pretrained(variant="base")
    img = load_image(IMAGE_ADDRESS)
    h, w = net.get_input_spec()["image"][0][2:]
    _, _, label, _ = DETRApp(net, h, w).predict(img, threshold=0.7)
    assert set(label.numpy()) == EXPECTED_OUTPUT


def test_cli_from_pretrained() -> None:
    args = get_model_cli_parser(RF_DETR).parse_args([])
    assert model_from_cli_args(RF_DETR, args) is not None


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_variant_from_pretrained(variant: str) -> None:
    net = RF_DETR.from_pretrained(variant=variant)
    assert net is not None
    assert net.variant == variant


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_variant_input_spec(variant: str) -> None:
    net = RF_DETR.from_pretrained(variant=variant)
    input_spec = net.get_input_spec()
    shape = input_spec["image"][0]
    assert shape[2] == shape[3] == VARIANT_RESOLUTION[variant]


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
