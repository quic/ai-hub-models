# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import qai_hub as hub

from qai_hub_models.utils.qai_hub_helpers import extract_job_options


def assert_options_eq(options: str, options_dict: dict[str, str | bool]):
    assert extract_job_options(MagicMock(spec=hub.Job, options=options)) == options_dict


def test_extract_job_options():
    assert_options_eq("", {})
    assert_options_eq("--boolean_flag", {"boolean_flag": True})
    assert_options_eq(
        "--blah text --boolean_flag", {"blah": "text", "boolean_flag": True}
    )
    assert_options_eq(
        "--boolean_flag --dict-input='blah=true;x=y'",
        {"dict-input": "blah=true;x=y", "boolean_flag": True},
    )
    assert_options_eq("--dict-input 'blah=true;x=y'", {"dict-input": "blah=true;x=y"})
    assert_options_eq('--dict-input "blah=true;x=y"', {"dict-input": "blah=true;x=y"})
    assert_options_eq('--dict-input="blah=true;x=y"', {"dict-input": "blah=true;x=y"})
    assert_options_eq("--dict-input=blah=true;x=y", {"dict-input": "blah=true;x=y"})
