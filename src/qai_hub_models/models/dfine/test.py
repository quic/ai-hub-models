# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.dfine.app import DFineApp
from qai_hub_models.models.dfine.demo import main as demo_main
from qai_hub_models.models.dfine.model import (
    IMAGE_ADDRESS,
    DFine,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import load_image

# Classes the default (nano-coco) variant detects on the demo image:
# 15 (cat), 57 (couch), 65 (remote).
EXPECTED_OUTPUT = {15, 57, 65}


def test_task() -> None:
    net = DFine.from_pretrained()
    img = load_image(IMAGE_ADDRESS)
    _, _, label, _ = DFineApp(net, 640, 640).predict(img)
    assert set(label.numpy()) == EXPECTED_OUTPUT


def test_cli_from_pretrained() -> None:
    args = get_model_cli_parser(DFine).parse_args([])
    assert model_from_cli_args(DFine, args) is not None


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
