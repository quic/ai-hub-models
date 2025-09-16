# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import functools

import requests

from qai_hub_models._version import __version__ as qaihm_version


class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


class QAIHMVersion:
    @classproperty
    @functools.cache
    def all(cls) -> list[str]:
        """
        Fetches and prints all released versions of ai hub models from PyPi.

        Returns:
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
        """
        Get the version string for the latest released version of AI Hub Models."""
        return f"v{QAIHMVersion.all[0]}"

    @classproperty
    def current_tag(cls) -> str:
        """
        Get the version string for the latest released version of AI Hub Models."""
        return f"v{qaihm_version}"

    @staticmethod
    def tag_from_string(version_str: str) -> str:
        """
        Gets the version tag from the given version string.
        """
        if not version_str.startswith("v"):
            if version_str.lower() == "latest":
                return QAIHMVersion.latest_tag
            return f"v{version_str}"
        return version_str
