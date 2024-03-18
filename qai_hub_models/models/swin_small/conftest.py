# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from unittest.mock import patch

import pytest

from qai_hub_models.models.swin_small import Model


@pytest.fixture(autouse=True)
def mock_from_pretrained():
    """
    Model.from_pretrained() can be slow. Invoke it once and cache it so all invocations
    across all tests return the cached instance of the model.
    """
    mock = patch(
        "qai_hub_models.models.swin_small.Model.from_pretrained",
        return_value=Model.from_pretrained(),
    )
    mock.start()
