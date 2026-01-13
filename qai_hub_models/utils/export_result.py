# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path

import qai_hub as hub

from qai_hub_models.configs.tool_versions import ToolVersions


@dataclass
class ExportResult:
    compile_job: hub.CompileJob | None = None
    quantize_job: hub.QuantizeJob | None = None
    profile_job: hub.ProfileJob | None = None
    inference_job: hub.InferenceJob | None = None
    link_job: hub.LinkJob | None = None

    # Unset for models with multiple components; see CollectionExportResult.
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None


@dataclass
class CollectionExportResult:
    components: dict[str, ExportResult]
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
