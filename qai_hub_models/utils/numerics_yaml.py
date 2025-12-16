# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from qai_hub_models.configs.numerics_yaml import (
    QAIHMModelNumerics,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.numerics_diff import NumericsDiff
from qai_hub_models.utils.testing_async_utils import get_accuracy_metadata_columns


class AccuracyMetadata(NamedTuple):
    dataset_name: str
    dataset_link: str
    split_description: str
    metric_name: str
    metric_unit: str | None
    metric_description: str
    metric_min: float | None
    metric_max: float | None
    metric_threshold: float | None
    num_samples: int


def _extract_metadata(df: pd.DataFrame) -> AccuracyMetadata | str:
    values_dict = {}
    for column_name in get_accuracy_metadata_columns():
        column_values = df[column_name].unique()
        if len(column_values) != 1:
            return f"Expected exactly one column value for field {column_name}, got {column_values.tolist()}."

        value = column_values[0]
        if isinstance(value, float) and np.isnan(value):
            value = None
        if column_name in ["metric_min", "metric_max", "metric_threshold"]:
            value = float(value) if value is not None else None
        values_dict[column_name] = value
    return AccuracyMetadata(**values_dict)


def create_numerics_struct(
    model_name: str,
    accuracy_df: pd.DataFrame,
    chipset_registry: dict[str, ScorecardDevice],
) -> QAIHMModelNumerics | None:
    model_df = accuracy_df[accuracy_df.model_id == model_name]
    if model_df.empty:
        print(f"Model {model_name} has no accuracy data. Skipping.")
        return None
    final_data: list[QAIHMModelNumerics.MetricDetails] = []
    for _, metric_df in model_df.groupby("metric_name"):
        device_metric: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ] = {}
        accuracy_metadata_or_error = _extract_metadata(metric_df)
        if isinstance(accuracy_metadata_or_error, str):
            print(
                f"Incomplete metadata for model {model_name}: {accuracy_metadata_or_error}"
            )
            return None
        accuracy_metadata = accuracy_metadata_or_error

        if pd.isna(metric_df["Torch Accuracy"]).any():
            print(
                f"Model {model_name} is missing torch accuracy for some rows. Skipping.",
            )
            return None
        if len(metric_df["Torch Accuracy"].unique()) != 1:
            print(
                f"Unexpected duplicate values for torch accuracy for model {model_name}. Skipping.",
            )
            return None
        torch_accuracy = metric_df["Torch Accuracy"].unique()[0]
        for chipset in metric_df.chipset.unique():
            chipset_df = metric_df[metric_df.chipset == chipset]
            metric_data_dict: dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ] = {}
            for _, row in chipset_df.iterrows():
                device_accuracy = row["Device Accuracy"]
                if pd.isna(device_accuracy):
                    continue
                precision = Precision.parse(row.precision)
                runtime = ScorecardProfilePath(row.runtime)
                if not runtime.include_in_perf_yaml:
                    continue
                if precision not in metric_data_dict:
                    metric_data_dict[precision] = {}
                metric_data_dict[precision][runtime] = QAIHMModelNumerics.DeviceDetails(
                    partial_metric=float(device_accuracy)
                )
            device_metric[chipset_registry[chipset]] = metric_data_dict

        final_data.append(
            QAIHMModelNumerics.MetricDetails(
                dataset_name=accuracy_metadata.dataset_name,
                dataset_link=accuracy_metadata.dataset_link,
                dataset_split_description=accuracy_metadata.split_description,
                metric_name=accuracy_metadata.metric_name,
                metric_description=accuracy_metadata.metric_description,
                metric_unit=accuracy_metadata.metric_unit or "",
                metric_range=QAIHMModelNumerics.Range(
                    min=accuracy_metadata.metric_min,
                    max=accuracy_metadata.metric_max,
                ),
                metric_fp_vs_device_enablement_threshold=accuracy_metadata.metric_threshold,
                num_partial_samples=accuracy_metadata.num_samples,
                partial_torch_metric=torch_accuracy,
                device_metric=device_metric,
            )
        )
    if len(final_data) == 0:
        print(f"Model {model_name} has no torch metric present. Skipping")
        return None
    return QAIHMModelNumerics(metrics=final_data)


def create_numerics_yaml(
    model_name: str,
    accuracy_df: pd.DataFrame,
    chipset_registry: dict[str, ScorecardDevice],
    numerics_diff: NumericsDiff | None = None,
) -> QAIHMModelNumerics | None:
    existing_struct = QAIHMModelNumerics.from_model(model_name, not_exists_ok=True)
    numerics_struct = create_numerics_struct(model_name, accuracy_df, chipset_registry)
    if numerics_diff is not None:
        numerics_diff.update_summary(model_name, existing_struct, numerics_struct)
    return numerics_struct or None


def get_chipset_registry() -> dict[str, ScorecardDevice]:
    return {device.chipset: device for device in ScorecardDevice.all_devices()}
