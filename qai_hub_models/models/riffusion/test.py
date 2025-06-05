# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models._shared.stable_diffusion.test_utils import (
    export_for_component,
)
from qai_hub_models.models.riffusion.demo import main as demo_main
from qai_hub_models.models.riffusion.export import export_model
from qai_hub_models.models.riffusion.model import Riffusion


def test_from_precompiled() -> None:
    Riffusion.from_precompiled()


@pytest.mark.skip("#105 move slow_cloud and slow tests to nightly.")
@pytest.mark.slow_cloud
def test_export() -> None:
    export_for_component(export_model, "TextEncoder_Quantized")


@pytest.mark.skip("#105 move slow_cloud and slow tests to nightly.")
@pytest.mark.slow_cloud
def test_demo() -> None:
    demo_main(is_test=True)
