# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import os
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

import requests
from packaging.version import Version
from packaging.version import parse as parse_version

from qai_hub_models_cli._internal.utils import use_internal_releases
from qai_hub_models_cli._version import __version__
from qai_hub_models_cli.common import CACHE_DIR
from qai_hub_models_cli.envvars import (
    FORCE_VERSION_ENVVAR,
    USE_INTERNAL_RELEASES_ENVVAR,
)
from qai_hub_models_cli.errors import UnsupportedVersionError

CURRENT_VERSION = parse_version(os.environ.get(FORCE_VERSION_ENVVAR, __version__))

# the minimum AI Hub Models version supported by the CLI
MIN_SUPPORTED_VERSION = Version("0.44.0")

# the version where manifest files were first published
MIN_MANIFEST_VERSION = Version("0.52.0")

# the version where internal manifest files were first published
MIN_INTERNAL_REGISTRY_VERSION = Version("0.56.0")

# the version where the manifest gained model-filter and runtime-detail fields
MIN_MODEL_FILTER_VERSION = Version("0.56.0")

PYPI_VERSIONS_URL = "https://pypi.org/pypi/qai-hub-models/json"


def _pip_package_name() -> str:
    """Return the pip package name to use in upgrade instructions."""
    try:
        pkg_version("qai_hub_models")
        return "qai-hub-models"
    except PackageNotFoundError:
        return "qai-hub-models-cli"


_VERSIONS_CACHE = CACHE_DIR / "published-versions.txt"
_CACHE_MAX_AGE_SECONDS = 3 * 24 * 60 * 60  # 3 days


def _fetch_published_versions() -> list[Version]:
    """Fetch all published versions from PyPI. Returns sorted newest-first."""
    resp = requests.get(PYPI_VERSIONS_URL, timeout=10)
    resp.raise_for_status()
    releases = resp.json().get("releases", {})
    versions = {parse_version(v) for v in releases}
    versions.add(CURRENT_VERSION)
    return sorted(versions, reverse=True)


@functools.lru_cache(maxsize=1)
def get_published_versions(force_refresh: bool = False) -> list[Version]:
    """Return all published versions sorted newest-first, using a disk cache.

    The cache is refreshed from PyPI at most once every
    ``_CACHE_MAX_AGE_SECONDS`` seconds.  When *force_refresh* is True,
    always queries PyPI and updates the cache.
    """
    if not force_refresh:
        try:
            if _VERSIONS_CACHE.exists():
                age = time.time() - _VERSIONS_CACHE.stat().st_mtime
                if age < _CACHE_MAX_AGE_SECONDS:
                    lines = _VERSIONS_CACHE.read_text().strip().splitlines()
                    versions = [parse_version(line) for line in lines if line]
                    # Cache is stale if the installed version is newer
                    # than everything in the cache (e.g. user just upgraded)
                    # and isn't a dev release.
                    if versions and not (
                        not CURRENT_VERSION.is_devrelease
                        and versions[0] < CURRENT_VERSION
                    ):
                        return versions
        except (OSError, ValueError):
            pass

    versions = _fetch_published_versions()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _VERSIONS_CACHE.write_text("\n".join(str(v) for v in versions))
    except OSError:
        pass
    return versions


@functools.lru_cache(maxsize=1)
def get_supported_versions(force_refresh: bool = False) -> list[Version]:
    """Return supported versions sorted newest-first.

    Filters ``get_published_versions`` to the range
    [``MIN_SUPPORTED_VERSION``, ``CURRENT_VERSION``], stopping early
    since the input is already sorted descending.
    """
    versions = get_published_versions(force_refresh=force_refresh)
    # Find start: first version <= CURRENT_VERSION (from the front)
    start = 0
    while start < len(versions) and versions[start] > CURRENT_VERSION:
        start += 1
    # Find end: last version >= MIN_SUPPORTED_VERSION (from the back)
    end = len(versions)
    while end > start and versions[end - 1] < MIN_SUPPORTED_VERSION:
        end -= 1
    return versions[start:end]


def print_upgrade_notice(force_refresh: bool = False) -> None:
    """Print an upgrade notice if a newer version is available on PyPI.

    Uses the disk-cached version list by default.  When *force_refresh*
    is True, always queries PyPI regardless of cache age.
    """
    try:
        latest = get_published_versions(force_refresh=force_refresh)[0]
        if latest > CURRENT_VERSION:
            print(
                f"\nA newer version is available ({latest}). We recommend you use the latest CLI, even to access older releases."
                f"\nUpgrade: pip install --upgrade {_pip_package_name()}"
            )
    except (requests.RequestException, OSError, ValueError):
        pass


def normalize_version(version: str) -> str:
    """Lowercase and strip leading 'v' from a version string."""
    return version.lower().removeprefix("v")


def feature_supported(version: Version, min_version: Version) -> bool:
    """Whether a feature introduced in *min_version* is available for *version*.

    Normally this is just ``version >= min_version``. On a dev/source install
    the default ``version`` is the synthetic dev ``CURRENT_VERSION`` (e.g.
    ``0.56.1.dev77+g...``), which sorts below the next release even though the
    source tree already has the feature. Treat that case as supported so
    new features aren't gated off when running from a source checkout.
    """
    if version == CURRENT_VERSION and CURRENT_VERSION.is_devrelease:
        return True
    return version >= min_version


def version_flag(version: Version) -> str:
    """Return ``"-v <version>"`` for a non-current release, else ``""``.

    Sample commands shown to the user must carry ``-v`` when browsing a release
    other than the installed one, or following them would silently retarget the
    installed version. Pass the result as ``format_command_sections``'s version_suffix.
    """
    return f"-v {version}" if version != CURRENT_VERSION else ""


def verify_not_dev_release(version: Version) -> None:
    """Raise if *version* is a dev release (e.g. ``0.45.0.dev1``)."""
    if version.is_devrelease:
        raise UnsupportedVersionError(
            f"Version {version} is a dev version and has no published assets. "
            f"Provide a release version with -v (e.g. -v 0.45.0)."
        )


def verify_version_supported(
    version: Version, verify_manifest_supported: bool = False
) -> None:
    """
    Validate that a requested version is readable by this version of the CLI.

    Ensures *version* is at or above ``MIN_SUPPORTED_VERSION``,
    which is the earliest release that published per-device assets to S3.

    Also ensures *version* does not exceed the currently installed package
    version, since assets for a newer release may rely on schema or runtime
    changes not present in the current install.

    Parameters
    ----------
    version
        AI Hub Models version.
    verify_manifest_supported
        If True, also require *version* to be at or above the first release
        that published manifest files (``MIN_MANIFEST_VERSION``).

    Raises
    ------
    UnsupportedVersionError
        If *version* is below the minimum, above the installed version,
        or not a published release.
    """
    if version < MIN_SUPPORTED_VERSION:
        raise UnsupportedVersionError(
            f"Version {version} is not supported. Minimum supported version is {MIN_SUPPORTED_VERSION}.\n"
            "Run `qai-hub-models versions` to see all supported versions."
        )

    if use_internal_releases() and version < MIN_INTERNAL_REGISTRY_VERSION:
        raise UnsupportedVersionError(
            f"Version {version} does not have an internal release. Unset {USE_INTERNAL_RELEASES_ENVVAR} to use the public release instead."
            f" An internal release is available for v{MIN_INTERNAL_REGISTRY_VERSION} and above."
        )

    if verify_manifest_supported and version < MIN_MANIFEST_VERSION:
        raise UnsupportedVersionError(
            f"Version {version} does not support this operation. Update to version {MIN_MANIFEST_VERSION} or higher."
        )

    if version > CURRENT_VERSION and not CURRENT_VERSION.is_devrelease:
        raise UnsupportedVersionError(
            f"Version {version} is newer than the installed version ({__version__}). "
            f"Upgrade the package or use -v with an older version.\n"
            "Run `qai-hub-models versions` to see all supported versions."
        )

    if CURRENT_VERSION.is_devrelease:
        # dev releases are allowed to look forward
        return

    published = get_published_versions()
    if version not in published:
        recent = ", ".join(str(v) for v in published[:5])
        raise UnsupportedVersionError(
            f"Version {version} is not a published release.\n"
            f"Available versions: {recent}, ...\n"
            "Run `qai-hub-models versions` to see all supported versions."
        )
