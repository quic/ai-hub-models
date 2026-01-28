# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import functools
from importlib.metadata import version
from typing import Any

import requests
from packaging.version import Version

from qai_hub_models._version import __version__ as qaihm_version


def ensure_supported_version(
    package: str,
    min_version: str | None = None,
    below_version: str | None = None,
) -> None:
    """
    Ensure a package version is within the supported range.

    Parameters
    ----------
    package
        Package name (e.g., "torch", "transformers").
    min_version
        Minimum supported version (inclusive: >=).
    below_version
        Must be below this version (exclusive: <).

    Raises
    ------
    ValueError
        If the package version is outside the supported range.

    Examples
    --------
    >>> ensure_supported_version("torch", below_version="2.9")
    >>> ensure_supported_version("transformers", min_version="4.48")
    >>> ensure_supported_version("aimet_onnx", min_version="2.17", below_version="2.21")
    """
    current_version = Version(version(package))

    if min_version is not None and current_version < Version(min_version):
        raise ValueError(
            f"{package} {current_version} is not supported. "
            f"Please run `pip install {package}=={min_version}`."
        )

    if below_version is not None and current_version >= Version(below_version):
        raise ValueError(
            f"{package} {current_version} is not supported. "
            f"Please use {package} < {below_version}."
        )


class classproperty:
    def __init__(self, func: Any) -> None:
        self.fget = func

    def __get__(self, instance: Any, owner: type) -> Any:
        return self.fget(owner)


class QAIHMVersion:
    @classproperty
    @functools.cache
    def all(cls) -> list[str]:
        """
        Fetches and prints all released versions of ai hub models from PyPi.

        Returns
        -------
        list[str]
            A list of all releases, latest first.
        """
        url = "https://pypi.org/pypi/qai_hub_models/json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()
        releases = data.get("releases", {})
        return list(releases.keys())[::-1]

    @classproperty
    def latest_tag(cls) -> str:
        """Get the version string for the latest released version of AI Hub Models."""
        return f"v{QAIHMVersion.all[0]}"

    @classproperty
    def current_tag(cls) -> str:
        """Get the version string for the latest released version of AI Hub Models."""
        return f"v{qaihm_version}"

    @staticmethod
    def tag_from_string(version_str: str) -> str:
        """Gets the version tag from the given version string."""
        if not version_str.startswith("v"):
            if version_str.lower() == "latest":
                return QAIHMVersion.latest_tag
            return f"v{version_str}"
        return version_str
