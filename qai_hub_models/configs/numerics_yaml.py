# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


def get_numerics_yaml_path(model_id: str) -> Path:
    return QAIHM_MODELS_ROOT / model_id / "numerics.yaml"


class QAIHMModelNumerics(BaseQAIHMConfig):
    class DeviceDetails(BaseQAIHMConfig):
        partial_metric: float

    class MetricDetails(BaseQAIHMConfig):
        dataset_name: str
        dataset_link: str
        dataset_split_description: str
        metric_name: str
        metric_description: str
        metric_unit: str
        num_partial_samples: int
        partial_torch_metric: float
        device_metric: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ] = Field(default_factory=dict)

    metrics: list[MetricDetails]

    def to_model_yaml(self, model_id: str) -> Path:
        path = get_numerics_yaml_path(model_id)
        self.to_yaml(path)
        return path

    @classmethod
    def from_model(
        cls: type[QAIHMModelNumerics], model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelNumerics | None:
        numerics_path = get_numerics_yaml_path(model_id)
        if not_exists_ok and not os.path.exists(numerics_path):
            return None
        return cls.from_yaml(numerics_path)
