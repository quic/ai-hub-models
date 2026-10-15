# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from packaging.version import Version

from qai_hub_models._version import __version__
from qai_hub_models.utils.version_helpers import get_tested_torch_version


def test_version() -> None:
    Version(__version__)


def test_torch_requirement_has_upper_bound() -> None:
    tested = get_tested_torch_version()
    assert tested is not None, (
        "requirements.txt must pin an upper bound (<= or ==) on torch; it is "
        "reported to users as the latest tested torch version."
    )
    Version(tested)
