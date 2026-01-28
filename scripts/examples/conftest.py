# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--on-device", action="store_true", default=False, help="Run on-device tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "on_device: Tests running Hub inference jobs")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--on-device"):
        # --on-device given in cli: do not skip on_device tests
        return
    skip_on_device = pytest.mark.skip(reason="need --on-device option to run")
    for item in items:
        if "on_device" in item.keywords:
            item.add_marker(skip_on_device)
