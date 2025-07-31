# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "compile: Run compile tests.")
    config.addinivalue_line("markers", "quantize: Run quantize tests.")
    config.addinivalue_line("markers", "profile: Run profile tests.")
    config.addinivalue_line("markers", "inference: Run inference tests.")
    config.addinivalue_line("markers", "trace: Run trace accuracy tests.")


def pytest_collection_modifyitems(items, config):
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("unmarked")
