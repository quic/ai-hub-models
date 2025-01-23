# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import (
    for_each_scorecard_path_and_device,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_summary import (
    ModelCompileSummary,
    ModelInferenceSummary,
    ModelPerfSummary,
    ScorecardJobTypeVar,
)


@dataclass
class ResultsSpreadsheet(list):
    has_compile_jobs: bool = True
    has_profile_jobs: bool = True
    has_inference_jobs: bool = True

    @dataclass
    class Entry:
        model_id: str
        component_id: str | None
        quantized: bool
        chipset: str
        runtime: TargetRuntime
        compile_status: str
        compile_url: str | None
        profile_status: str
        profile_url: str | None
        inference_time: float | None
        NPU: int | None
        GPU: int | None
        CPU: int | None
        inference_status: str
        inference_url: str | None

        @property
        def model_component_id(self) -> str:
            return self.model_id + (
                f"::{self.component_id}" if self.component_id else ""
            )

    def to_csv(
        self, path: str | os.PathLike, combine_model_and_component_id: bool = True
    ):
        # Get CSV fields
        field_names = [field.name for field in fields(ResultsSpreadsheet.Entry)]

        # Remove fields that are not applicable
        fields_to_remove: list[str] = []
        if not self.has_compile_jobs:
            fields_to_remove.extend(["compile_status", "compile_url"])
        if not self.has_profile_jobs:
            fields_to_remove.extend(
                ["profile_status", "profile_url", "inference_time", "NPU", "GPU", "CPU"]
            )
        if not self.has_inference_jobs:
            fields_to_remove.extend(["inference_status", "inference_url"])
        if combine_model_and_component_id:
            fields_to_remove.extend(["component_id"])
        for field in fields_to_remove:
            field_names.remove(field)

        # Save CSV
        with open(path, "w") as csvfile:
            scorecard_csv = csv.writer(csvfile)

            # Header
            scorecard_csv.writerow(field_names)

            # Combined ID should be included instead of separate columns for model & component
            if combine_model_and_component_id:
                field_names.remove("model_id")
                field_names.insert(0, "model_component_id")

            def _entry_str(val):
                if val is None:
                    return ""
                if isinstance(val, bool):
                    return "YES" if val else "NO"
                return str(val)

            # Rows
            for entry in self:
                scorecard_csv.writerow(
                    [_entry_str(getattr(entry, name)) for name in field_names]
                )

    @dataclass
    class ModelInfo:
        model_id: str
        components: list[str] | None
        quantized: bool

    def append_model_summary_entries(
        self,
        is_quantized: bool,
        compile_summary: ModelCompileSummary | None = None,
        profile_summary: ModelPerfSummary | None = None,
        inference_summary: ModelInferenceSummary | None = None,
    ):
        self.extend(
            ResultsSpreadsheet.get_model_summary_entries(
                is_quantized, compile_summary, profile_summary, inference_summary
            )
        )

    @staticmethod
    def get_model_summary_entries(
        is_quantized: bool,
        compile_summary: ModelCompileSummary | None = None,
        profile_summary: ModelPerfSummary | None = None,
        inference_summary: ModelInferenceSummary | None = None,
    ) -> list[ResultsSpreadsheet.Entry]:
        entries: list[ResultsSpreadsheet.Entry] = []

        # Validate input summaries match
        base_summary = compile_summary or profile_summary or inference_summary
        if not base_summary:
            return []
        for summary in [compile_summary, profile_summary, inference_summary]:
            if summary and not base_summary.is_same_model(summary):
                raise ValueError("Summaries do not point to the same model definition.")

        # Extract model and component ids
        model_id = base_summary.model_id
        component_ids = base_summary.component_ids

        # Create empty summaries if a relevant summary is not passed.
        # Empty summaries will always return "skipped" jobs when queried for runs.
        compile_summary = compile_summary or ModelCompileSummary(
            model_id, None, {x: {} for x in component_ids}
        )
        profile_summary = profile_summary or ModelPerfSummary(
            model_id, None, {x: {} for x in component_ids}
        )
        inference_summary = inference_summary or ModelInferenceSummary(
            model_id, None, {x: {} for x in component_ids}
        )

        def create_entry(path: ScorecardProfilePath, device: ScorecardDevice):
            for component_id in component_ids:
                # Get job for this path + device + component combo
                compile_job = compile_summary.get_run(
                    device, path.compile_path, component_id
                )
                profile_job = profile_summary.get_run(device, path, component_id)
                inference_job = inference_summary.get_run(device, path, component_id)

                def _get_url_and_status(
                    sjob: ScorecardJobTypeVar,
                ) -> tuple[str, str | None]:
                    return (
                        sjob.job_status
                        + (f" ({sjob.status_message})" if sjob.status_message else ""),
                        sjob.job.url if not sjob.skipped else None,
                    )

                # Job status
                compile_status, compile_url = _get_url_and_status(compile_job)
                profile_status, profile_url = _get_url_and_status(profile_job)
                inference_status, inference_url = _get_url_and_status(inference_job)

                # Profile job results
                if profile_job.success:
                    inference_time = profile_job.inference_time / 1000  # type: ignore
                    NPU = profile_job.npu
                    GPU = profile_job.gpu
                    CPU = profile_job.cpu
                else:
                    inference_time = None
                    NPU = None
                    GPU = None
                    CPU = None

                # Create Entry
                entry = ResultsSpreadsheet.Entry(
                    model_id=model_id,
                    component_id=component_id if component_id != model_id else None,
                    quantized=is_quantized,
                    chipset=device.chipset,
                    runtime=path.runtime,
                    compile_status=compile_status,
                    compile_url=compile_url,
                    profile_status=profile_status,
                    profile_url=profile_url,
                    inference_time=inference_time,
                    NPU=NPU,
                    GPU=GPU,
                    CPU=CPU,
                    inference_status=inference_status,
                    inference_url=inference_url,
                )
                entries.append(entry)

        for_each_scorecard_path_and_device(
            is_quantized,
            ScorecardProfilePath,
            create_entry,
            exclude_devices=[cs_universal],
        )

        return entries
