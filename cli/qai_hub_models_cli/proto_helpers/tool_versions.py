# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Helpers for the ``ToolVersions`` proto and the ``-s/--sdk-version`` filter."""

from __future__ import annotations

from qai_hub_models_cli.errors import CLIUserError
from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions

# Tool version proto fields paired with their display labels, in display order.
_TOOL_VERSION_LABELS: list[tuple[str, str]] = [
    ("qairt", "QAIRT"),
    ("geniex", "GenieX"),
    ("onnx", "ONNX"),
    ("onnx_runtime", "ONNX Runtime"),
    ("tflite", "TFLite"),
    ("litert", "LiteRT"),
    ("ai_hub_models", "AI Hub Models"),
]

# Accepted tool names for an SDK version filter, mapped to their proto field.
# Includes the field name and a normalized form of the display label.
_SDK_FILTER_TOOLS: dict[str, str] = {
    **{field: field for field, _ in _TOOL_VERSION_LABELS},
    **{label.lower(): field for field, label in _TOOL_VERSION_LABELS},
}

# Canonical tool names accepted by the ``-s/--sdk-version`` filter, in display
# order. The display labels are also accepted (case-insensitively).
SDK_FILTER_TOOL_NAMES: list[str] = [field for field, _ in _TOOL_VERSION_LABELS]


def format_tool_versions(tool_versions: ToolVersions) -> str:
    """Format the set tool versions as a comma-separated ``Label X.Y`` string."""
    parts = [
        f"{label} {value}"
        for field, label in _TOOL_VERSION_LABELS
        if (value := getattr(tool_versions, field))
    ]
    return ", ".join(parts) if parts else "—"


def validate_sdk_tools(sdk_versions: dict[str, str]) -> None:
    """Raise ``CLIUserError`` if any key in *sdk_versions* is not a known SDK tool.

    Unlike :func:`tool_versions_match` (which only validates when an asset is
    actually compared), this checks the tool names eagerly so a typo is reported
    even when no asset matches.
    """
    for tool in sdk_versions:
        if tool not in _SDK_FILTER_TOOLS:
            valid = ", ".join(SDK_FILTER_TOOL_NAMES)
            raise CLIUserError(f"Unknown SDK tool {tool!r}. Valid tools: {valid}.")


def tool_versions_match(
    tool_versions: ToolVersions, sdk_versions: dict[str, str]
) -> bool:
    """Whether *tool_versions* satisfies every entry in *sdk_versions*.

    Parameters
    ----------
    tool_versions
        The asset's tool versions to test.
    sdk_versions
        Map of tool name (``qairt``, ``onnx``, ``onnx_runtime``, ``tflite``,
        ``litert``, ``ai_hub_models``, or a display label) to a version
        substring, as produced by :func:`parse_sdk_version_filters`. Each tool
        name is resolved to its proto field here, and its version is matched as
        a case-insensitive substring of the asset's value for that tool.

    Returns
    -------
    bool
        True if, for every entry, the asset's named tool version contains the
        requested substring.

    Raises
    ------
    CLIUserError
        If a tool name does not match a known tool.
    """
    for tool, version in sdk_versions.items():
        field = _SDK_FILTER_TOOLS.get(tool)
        if field is None:
            valid = ", ".join(SDK_FILTER_TOOL_NAMES)
            raise CLIUserError(f"Unknown SDK tool {tool!r}. Valid tools: {valid}.")
        if not version:
            # An empty version is a substring of everything, so without this
            # guard it would silently pass every asset.
            raise CLIUserError(
                f"SDK version filter for {tool!r} is empty. "
                "Use 'tool=version' syntax, e.g. 'litert=1.4.4'."
            )
        if version not in getattr(tool_versions, field).lower():
            return False
    return True
