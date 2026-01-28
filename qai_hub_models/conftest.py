# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "compile: Run compile tests.")
    config.addinivalue_line("markers", "quantize: Run quantize tests.")
    config.addinivalue_line("markers", "profile: Run profile tests.")
    config.addinivalue_line("markers", "inference: Run inference tests.")
    config.addinivalue_line("markers", "trace: Run trace accuracy tests.")


def pytest_collection_modifyitems(
    items: list[pytest.Item], config: pytest.Config
) -> None:
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("unmarked")
