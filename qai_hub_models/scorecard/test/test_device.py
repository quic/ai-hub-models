# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.scorecard.device import ScorecardDevice


def test_exactly_one_default():
    num_defaults = 0
    for device in ScorecardDevice._registry.values():
        if device.is_default:
            num_defaults += 1
    assert num_defaults == 1, (
        f"Must be exactly one default device, found {num_defaults}"
    )
