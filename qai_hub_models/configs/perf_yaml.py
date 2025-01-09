# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.base_config import BaseQAIHMConfig


def bytes_to_mb(num_bytes: int) -> int:
    return round(num_bytes / (1 << 20))


class QAIHMModelPerf:
    """Class to read model perf.yaml"""

    ###
    # Helper Struct Classes
    ###

    @dataclass
    class PerformanceDetails:
        job_id: str
        inference_time_microsecs: float
        peak_memory_bytes: tuple[int, int]  # min, max
        compute_unit_counts: dict[str, int]
        primary_compute_unit: str
        precision: str

        @staticmethod
        def from_dict(device_perf_details: dict) -> QAIHMModelPerf.PerformanceDetails:
            peak_memory = device_perf_details["estimated_peak_memory_range"]
            layer_info = device_perf_details["layer_info"]
            compute_unit_counts = {}
            for layer_name, count in layer_info.items():
                if "layers_on" in layer_name:
                    if count > 0:
                        compute_unit_counts[layer_name[-3:].upper()] = count

            return QAIHMModelPerf.PerformanceDetails(
                job_id=device_perf_details["job_id"],
                inference_time_microsecs=float(device_perf_details["inference_time"]),
                peak_memory_bytes=(peak_memory["min"], peak_memory["max"]),
                compute_unit_counts=compute_unit_counts,
                primary_compute_unit=device_perf_details["primary_compute_unit"],
                precision=device_perf_details["precision"],
            )

    @dataclass
    class LLMPerformanceDetails:
        time_to_first_token_range_secs: tuple[str, str]  # min, max
        tokens_per_second: float

        @staticmethod
        def from_dict(
            device_perf_details: dict,
        ) -> QAIHMModelPerf.LLMPerformanceDetails:
            ttftr = device_perf_details["time_to_first_token_range"]
            return QAIHMModelPerf.LLMPerformanceDetails(
                time_to_first_token_range_secs=(
                    # Original data is in microseconds
                    str(float(ttftr["min"]) * 1e-6),
                    str(float(ttftr["max"]) * 1e-6),
                ),
                tokens_per_second=device_perf_details["tokens_per_second"],
            )

    @dataclass
    class EvaluationDetails(BaseQAIHMConfig):
        name: str
        value: float
        unit: str

    @dataclass
    class DeviceDetails(BaseQAIHMConfig):
        name: str
        os: str
        form_factor: str
        os_name: str
        manufacturer: str
        chipset: str

    @dataclass
    class ProfilePerfDetails:
        path: ScorecardProfilePath
        perf_details: QAIHMModelPerf.PerformanceDetails | QAIHMModelPerf.LLMPerformanceDetails
        eval_details: QAIHMModelPerf.EvaluationDetails | None = None

        @staticmethod
        def from_dict(
            path: ScorecardProfilePath, perf_details_dict: dict
        ) -> QAIHMModelPerf.ProfilePerfDetails:
            perf_details: QAIHMModelPerf.LLMPerformanceDetails | QAIHMModelPerf.PerformanceDetails
            if llm_metrics := perf_details_dict.get("llm_metrics", None):
                perf_details = QAIHMModelPerf.LLMPerformanceDetails.from_dict(
                    llm_metrics
                )
            else:
                perf_details = QAIHMModelPerf.PerformanceDetails.from_dict(
                    perf_details_dict
                )

            if eval_metrics := perf_details_dict.get("evaluation_metrics", None):
                eval_details_data = (
                    QAIHMModelPerf.EvaluationDetails.get_schema().validate(eval_metrics)
                )
                eval_details = QAIHMModelPerf.EvaluationDetails.from_dict(
                    eval_details_data
                )
            else:
                eval_details = None

            return QAIHMModelPerf.ProfilePerfDetails(
                path=path, perf_details=perf_details, eval_details=eval_details
            )

    @dataclass
    class DevicePerfDetails:
        device: QAIHMModelPerf.DeviceDetails
        details_per_path: dict[ScorecardProfilePath, QAIHMModelPerf.ProfilePerfDetails]

        @staticmethod
        def from_dict(
            device: QAIHMModelPerf.DeviceDetails, device_runtime_details: dict
        ) -> QAIHMModelPerf.DevicePerfDetails:
            details_per_path = {}
            for profile_path in ScorecardProfilePath:
                if profile_path.long_name in device_runtime_details:
                    perf_details_dict = device_runtime_details[profile_path.long_name]
                    details_per_path[
                        profile_path
                    ] = QAIHMModelPerf.ProfilePerfDetails.from_dict(
                        profile_path, perf_details_dict
                    )
            return QAIHMModelPerf.DevicePerfDetails(
                device=device, details_per_path=details_per_path
            )

    @dataclass
    class ModelPerfDetails:
        model: str
        details_per_device: dict[str, QAIHMModelPerf.DevicePerfDetails]

        @staticmethod
        def from_dict(
            model: str, model_performance_metrics: list[dict]
        ) -> QAIHMModelPerf.ModelPerfDetails:
            details_per_device = {}
            for device_perf_details in model_performance_metrics:
                device_details_data = (
                    QAIHMModelPerf.DeviceDetails.get_schema().validate(
                        device_perf_details["reference_device_info"]
                    )
                )
                device_details = QAIHMModelPerf.DeviceDetails.from_dict(
                    device_details_data
                )
                details_per_device[
                    device_details.name
                ] = QAIHMModelPerf.DevicePerfDetails.from_dict(
                    device_details, device_perf_details
                )

            return QAIHMModelPerf.ModelPerfDetails(
                model=model, details_per_device=details_per_device
            )

    def __init__(self, perf_yaml_path: str | Path, model_name: str):
        self.model_name = model_name
        self.perf_yaml_path = perf_yaml_path
        self.per_model_details: dict[str, QAIHMModelPerf.ModelPerfDetails] = {}

        if os.path.exists(self.perf_yaml_path):
            self.perf_details = load_yaml(self.perf_yaml_path)
            all_models_and_perf = self.perf_details["models"]
            if not isinstance(all_models_and_perf, list):
                all_models_and_perf = [all_models_and_perf]

            for model_perf in all_models_and_perf:
                model_name = model_perf["name"]
                self.per_model_details[
                    model_name
                ] = QAIHMModelPerf.ModelPerfDetails.from_dict(
                    model_name, model_perf["performance_metrics"]
                )
