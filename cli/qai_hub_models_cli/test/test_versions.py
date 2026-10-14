# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from qai_hub_models_cli.cli import _run_versions, add_versions_parser
from qai_hub_models_cli.errors import UnsupportedVersionError
from qai_hub_models_cli.versions import (
    feature_supported,
    get_published_versions,
    get_supported_versions,
    normalize_version,
    print_upgrade_notice,
    verify_version_supported,
)

MOCK_VERSIONS = [
    Version(v) for v in ["0.49.0", "0.48.0", "0.47.0", "0.46.0", "0.45.0", "0.44.0"]
]


@contextmanager
def _mock_installed(ver: str) -> Any:
    """Mock both __version__ and CURRENT_VERSION in the versions module."""
    with (
        patch("qai_hub_models_cli.versions.__version__", ver),
        patch("qai_hub_models_cli.versions.CURRENT_VERSION", Version(ver)),
    ):
        yield


# ── normalize_version ────────────────────────────────────────────────


def test_normalize_version_strips_v_prefix() -> None:
    assert normalize_version("v0.45.0") == "0.45.0"


def test_normalize_version_no_prefix() -> None:
    assert normalize_version("0.45.0") == "0.45.0"


# ── verify_version_supported ────────────────────────────────────────────────


def _mock_pypi() -> Any:
    return patch(
        "qai_hub_models_cli.versions.get_published_versions",
        return_value=MOCK_VERSIONS,
    )


def test_verify_version_supported_below_floor() -> None:
    with (
        _mock_installed("0.99.0"),
        pytest.raises(UnsupportedVersionError, match="not supported"),
    ):
        verify_version_supported(Version("0.43.0"))


def test_verify_version_supported_above_installed() -> None:
    with (
        _mock_installed("0.45.0"),
        pytest.raises(UnsupportedVersionError, match="newer than the installed"),
    ):
        verify_version_supported(Version("0.46.0"))


def test_verify_version_supported_at_installed() -> None:
    with (
        _mock_installed("0.45.0"),
        _mock_pypi(),
    ):
        verify_version_supported(Version("0.45.0"))


def test_verify_version_supported_older_published() -> None:
    """Happy path: version is older than installed and is in the published list."""
    with (
        _mock_installed("0.49.0"),
        _mock_pypi(),
    ):
        verify_version_supported(Version("0.46.0"))


def test_verify_version_supported_not_published() -> None:
    with (
        _mock_installed("0.99.0"),
        _mock_pypi(),
        pytest.raises(UnsupportedVersionError, match="not a published release"),
    ):
        verify_version_supported(Version("0.50.0"))


def test_verify_version_supported_dev_install_skips_checks() -> None:
    # On a dev/source install the checks are skipped so any version is allowed,
    # even one newer than (the synthetic) installed version.
    with _mock_installed("0.56.1.dev77+gabc123"):
        verify_version_supported(Version("0.99.0"), verify_manifest_supported=True)


def test_feature_supported() -> None:
    # Released install: plain version comparison against the feature's minimum.
    with _mock_installed("0.57.0"):
        assert feature_supported(Version("0.57.0"), Version("0.57.0"))
        assert not feature_supported(Version("0.56.0"), Version("0.57.0"))
    # Dev/source install requesting the (dev) installed version: always supported.
    with _mock_installed("0.56.1.dev77+gabc123"):
        assert feature_supported(Version("0.56.1.dev77+gabc123"), Version("0.57.0"))
        # An explicit older release is still gated normally.
        assert not feature_supported(Version("0.56.0"), Version("0.57.0"))


# ── get_published_versions / get_supported_versions ─────────────


def _mock_pypi_response(versions: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"releases": {v: [] for v in versions}}
    return resp


def test_get_published_versions_sorted_descending(tmp_path: Path) -> None:
    pypi_versions = ["0.44.0", "0.46.0", "0.45.0"]
    with (
        _mock_installed("0.46.0"),
        patch("qai_hub_models_cli.versions._VERSIONS_CACHE", tmp_path / "none.txt"),
        patch("qai_hub_models_cli.versions.CACHE_DIR", tmp_path),
        patch(
            "qai_hub_models_cli.versions.requests.get",
            return_value=_mock_pypi_response(pypi_versions),
        ),
    ):
        result = get_published_versions()
    assert result == [Version("0.46.0"), Version("0.45.0"), Version("0.44.0")]


def test_get_supported_versions_caps_at_installed(tmp_path: Path) -> None:
    pypi_versions = ["0.43.0", "0.44.0", "0.45.0", "0.46.0", "0.47.0"]
    with (
        _mock_installed("0.46.0"),
        patch("qai_hub_models_cli.versions._VERSIONS_CACHE", tmp_path / "none.txt"),
        patch("qai_hub_models_cli.versions.CACHE_DIR", tmp_path),
        patch(
            "qai_hub_models_cli.versions.requests.get",
            return_value=_mock_pypi_response(pypi_versions),
        ),
    ):
        result = get_supported_versions()
    assert Version("0.47.0") not in result
    assert Version("0.46.0") in result
    assert Version("0.45.0") in result
    assert Version("0.43.0") not in result


# ── add_versions_parser ─────────────────────────────────────────────


def test_add_versions_parser_registers() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    versions_parser = add_versions_parser(subparsers)
    assert versions_parser is not None

    args = parser.parse_args(["versions"])
    assert hasattr(args, "func")
    assert args.quiet is False

    args_quiet = parser.parse_args(["versions", "-q"])
    assert args_quiet.quiet is True


# ── _run_versions ───────────────────────────────────────────────────


def _make_args(quiet: bool = False) -> argparse.Namespace:
    return argparse.Namespace(quiet=quiet)


MOCK_SUPPORTED = [v for v in MOCK_VERSIONS if v >= Version("0.44.0")]


def test_run_versions_quiet() -> None:
    with (
        patch(
            "qai_hub_models_cli.cli.get_supported_versions",
            return_value=MOCK_SUPPORTED,
        ),
        patch("builtins.print") as mock_print,
    ):
        _run_versions(_make_args(quiet=True))
    mock_print.assert_called_once()
    printed = mock_print.call_args[0][0]
    assert all(str(v) in printed for v in MOCK_SUPPORTED)


def test_run_versions_shows_installed_marker() -> None:
    with (
        patch("qai_hub_models_cli.cli.CURRENT_VERSION", Version("0.47.0")),
        patch(
            "qai_hub_models_cli.cli.get_supported_versions",
            return_value=MOCK_SUPPORTED,
        ),
        patch("qai_hub_models_cli.cli.print_upgrade_notice"),
        patch("builtins.print") as mock_print,
    ):
        _run_versions(_make_args())
    printed = " ".join(str(a) for call in mock_print.call_args_list for a in call[0])
    assert "0.47.0 (installed)" in printed


def test_run_versions_no_upgrade_notice_when_quiet() -> None:
    with (
        patch(
            "qai_hub_models_cli.cli.get_supported_versions",
            return_value=MOCK_SUPPORTED,
        ),
        patch("qai_hub_models_cli.cli.print_upgrade_notice") as mock_notice,
    ):
        _run_versions(_make_args(quiet=True))
    mock_notice.assert_not_called()


# ── print_upgrade_notice ─────────────────────────────────────────────


def test_print_upgrade_notice_when_newer_available() -> None:
    with (
        _mock_installed("0.45.0"),
        patch(
            "qai_hub_models_cli.versions.get_published_versions",
            return_value=MOCK_VERSIONS,
        ),
        patch("builtins.print") as mock_print,
    ):
        print_upgrade_notice()
    mock_print.assert_called_once()
    msg = mock_print.call_args[0][0]
    assert "0.49.0" in msg


def test_print_upgrade_notice_when_on_latest() -> None:
    with (
        _mock_installed("0.49.0"),
        patch(
            "qai_hub_models_cli.versions.get_published_versions",
            return_value=MOCK_VERSIONS,
        ),
        patch("builtins.print") as mock_print,
    ):
        print_upgrade_notice()
    mock_print.assert_not_called()


# ── get_published_versions disk cache ────────────────────────────────

MOCK_VERSIONS_TEXT = "\n".join(str(v) for v in MOCK_VERSIONS)


def test_get_published_versions_cache_miss(tmp_path: Path) -> None:
    """On cache miss, fetches from PyPI and writes cache file."""
    cache_file = tmp_path / "published-versions.txt"
    pypi_versions = ["0.44.0", "0.46.0", "0.45.0"]
    with (
        _mock_installed("0.46.0"),
        patch("qai_hub_models_cli.versions._VERSIONS_CACHE", cache_file),
        patch("qai_hub_models_cli.versions.CACHE_DIR", tmp_path),
        patch(
            "qai_hub_models_cli.versions.requests.get",
            return_value=_mock_pypi_response(pypi_versions),
        ),
    ):
        result = get_published_versions()
    assert result == [Version("0.46.0"), Version("0.45.0"), Version("0.44.0")]
    assert cache_file.exists()


def test_get_published_versions_cache_hit(tmp_path: Path) -> None:
    """Fresh cache file is read without calling PyPI."""
    cache_file = tmp_path / "published-versions.txt"
    cache_file.write_text(MOCK_VERSIONS_TEXT)
    with (
        _mock_installed("0.49.0"),
        patch("qai_hub_models_cli.versions._VERSIONS_CACHE", cache_file),
        patch("qai_hub_models_cli.versions.CACHE_DIR", tmp_path),
        patch("qai_hub_models_cli.versions.requests.get") as mock_get,
    ):
        result = get_published_versions()
    assert result == MOCK_VERSIONS
    mock_get.assert_not_called()


def test_get_published_versions_stale_cache(tmp_path: Path) -> None:
    """Stale cache triggers a refresh from PyPI."""
    cache_file = tmp_path / "published-versions.txt"
    cache_file.write_text("0.47.0\n0.46.0\n0.45.0\n0.44.0")
    old_time = time.time() - 4 * 24 * 60 * 60
    os.utime(cache_file, (old_time, old_time))
    pypi_versions = ["0.44.0", "0.45.0", "0.46.0", "0.47.0", "0.48.0"]
    with (
        _mock_installed("0.48.0"),
        patch("qai_hub_models_cli.versions._VERSIONS_CACHE", cache_file),
        patch("qai_hub_models_cli.versions.CACHE_DIR", tmp_path),
        patch(
            "qai_hub_models_cli.versions.requests.get",
            return_value=_mock_pypi_response(pypi_versions),
        ),
    ):
        result = get_published_versions()
    assert Version("0.48.0") in result
    assert cache_file.exists()


def test_get_published_versions_cache_stale_after_upgrade(tmp_path: Path) -> None:
    """Cache is invalidated when installed version is newer than cached max."""
    cache_file = tmp_path / "published-versions.txt"
    cache_file.write_text("0.47.0\n0.46.0\n0.45.0\n0.44.0")
    pypi_versions = ["0.44.0", "0.45.0", "0.46.0", "0.47.0", "0.48.0"]
    with (
        _mock_installed("0.48.0"),
        patch("qai_hub_models_cli.versions._VERSIONS_CACHE", cache_file),
        patch("qai_hub_models_cli.versions.CACHE_DIR", tmp_path),
        patch(
            "qai_hub_models_cli.versions.requests.get",
            return_value=_mock_pypi_response(pypi_versions),
        ) as mock_get,
    ):
        result = get_published_versions()
    mock_get.assert_called_once()
    assert Version("0.48.0") in result
