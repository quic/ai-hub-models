# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for parse_options.parse and its CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CI_DIR))
from parse_options import parse  # noqa: E402

_SCRIPT = _CI_DIR / "parse_options.py"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("", {}),
        ("qairt_version=2.45", {"qairt_version": "2.45"}),
        (
            "qairt_version=2.45,geniex_version=0.3.7",
            {"qairt_version": "2.45", "geniex_version": "0.3.7"},
        ),
        ("geniex_version=0.3.7", {"geniex_version": "0.3.7"}),
        # Order is irrelevant; whitespace and stray commas are tolerated.
        (
            "  geniex_version = 0.3.7 , , qairt_version = 2.45 ",
            {"qairt_version": "2.45", "geniex_version": "0.3.7"},
        ),
        # An explicitly empty qairt_version falls back to the default sentinel.
        ("qairt_version=", {}),
        ("genie_scope=perf", {"genie_scope": "perf"}),
    ],
)
def test_parse_ok(raw: str, want: dict[str, str]) -> None:
    expected = {
        "qairt_version": "qaihm_default",
        "geniex_version": "",
        "genie_scope": "",
    }
    expected.update(want)
    assert parse(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected_msg"),
    [
        # Bare (positional) values are not part of the grammar.
        ("2.45", "bare value"),
        ("qaihm_default", "bare value"),
        ("2.45,geniex_version=0.3.7", "bare value"),
        ("unknown_key=x", "Unknown option key"),
        ("qairt_version=2.45,qairt_version=2.50", "Duplicate option key"),
        ("geniex_version=0.3.7,geniex_version=0.3.8", "Duplicate option key"),
        # Bare SemVer required; leading v is rejected.
        ("geniex_version=v0.3.7", "must not start with 'v'"),
        ("genie_scope=compile_only", "genie_scope must be one of"),
    ],
)
def test_parse_errors(raw: str, expected_msg: str) -> None:
    with pytest.raises(ValueError, match=expected_msg):
        parse(raw)


def test_cli_writes_github_output(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            "qairt_version=2.45,geniex_version=0.3.7",
            "--github-output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    text = out.read_text()
    assert "qairt_version=2.45\n" in text
    assert "geniex_version=0.3.7\n" in text


def test_cli_writes_defaults_for_empty_input(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    res = subprocess.run(
        [sys.executable, str(_SCRIPT), "--raw", "", "--github-output", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    text = out.read_text()
    assert "qairt_version=qaihm_default\n" in text
    assert "geniex_version=\n" in text


def test_cli_emits_env_when_requested(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    env_file = tmp_path / "gh_env"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            "qairt_version=2.45,geniex_version=0.3.7",
            "--github-output",
            str(out),
            "--github-env",
            str(env_file),
            "--emit-env",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    env_text = env_file.read_text()
    assert "QAIHM_TEST_QAIRT_VERSION=2.45\n" in env_text
    assert "QAIHM_TEST_GENIEX_VERSION=0.3.7\n" in env_text


def test_cli_skips_env_by_default(tmp_path: Path) -> None:
    env_file = tmp_path / "gh_env"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            "qairt_version=2.45",
            "--github-env",
            str(env_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert not env_file.exists()


def test_cli_reports_error_and_exits_nonzero() -> None:
    res = subprocess.run(
        [sys.executable, str(_SCRIPT), "--raw", "unknown_key=x"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "Unknown option key" in res.stderr


def test_cli_rejects_newline_injection(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            "qairt_version=2.45\nqairt_version=eviltag",
            "--github-output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "newlines" in res.stderr
    assert not out.exists() or "eviltag" not in out.read_text()
