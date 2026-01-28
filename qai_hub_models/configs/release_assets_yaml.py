# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.scorecard.execution_helpers import cs_universal
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


class QAIHMModelReleaseAssets(BaseQAIHMConfig):
    """Schema for model release_assets.yaml files."""

    class AssetDetails(BaseQAIHMConfig):
        # Key for object in the AI Hub Models S3 Bucket
        # (see qai_hub_models/utils/_internal/aws.py)
        s3_key: str

        # Tool versions used to generate this asset.
        tool_versions: ToolVersions | None = None

    class PrecisionDetails(BaseQAIHMConfig):
        universal_assets: dict[
            ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails
        ] = Field(default_factory=dict)
        chipset_assets: dict[
            str,
            dict[ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails],
        ] = Field(default_factory=dict)

    precisions: dict[Precision, QAIHMModelReleaseAssets.PrecisionDetails] = Field(
        default_factory=dict
    )

    @property
    def empty(self) -> bool:
        return not self.precisions

    def add_asset(
        self,
        details: QAIHMModelReleaseAssets.AssetDetails,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
    ) -> None:
        if precision not in self.precisions:
            self.precisions[precision] = QAIHMModelReleaseAssets.PrecisionDetails()
        if (
            device != cs_universal
            and device.chipset not in self.precisions[precision].chipset_assets
        ):
            self.precisions[precision].chipset_assets[device.chipset] = {}
        if device != cs_universal:
            self.precisions[precision].chipset_assets[device.chipset][path] = details
        else:
            self.precisions[precision].universal_assets[path] = details

    def get_asset(
        self,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
    ) -> QAIHMModelReleaseAssets.AssetDetails | None:
        if precision not in self.precisions:
            return None
        if (
            device != cs_universal
            and device.chipset not in self.precisions[precision].chipset_assets
        ):
            return None
        if device != cs_universal:
            return self.precisions[precision].chipset_assets[device.chipset].get(path)
        return self.precisions[precision].universal_assets.get(path)

    @classmethod
    def from_model(
        cls: type[QAIHMModelReleaseAssets], model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelReleaseAssets:
        assets_path = QAIHM_MODELS_ROOT / model_id / "release-assets.yaml"
        if not_exists_ok and not os.path.exists(assets_path):
            return QAIHMModelReleaseAssets()
        return cls.from_yaml(assets_path)

    def to_model_yaml(self, model_id: str) -> Path:
        out = QAIHM_MODELS_ROOT / model_id / "release-assets.yaml"
        self.to_yaml(out)
        return out
