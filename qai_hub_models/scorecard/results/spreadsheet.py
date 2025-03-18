# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import cast

from qai_hub_models.configs.info_yaml import MODEL_DOMAIN, MODEL_TAG, MODEL_USE_CASE
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import (
    for_each_scorecard_path_and_device,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_summary import (
    ModelCompileSummary,
    ModelInferenceSummary,
    ModelPerfSummary,
    ModelPrecisionCompileSummary,
    ModelPrecisionInferenceSummary,
    ModelPrecisionPerfSummary,
    ModelPrecisionQuantizeSummary,
    ModelQuantizeSummary,
    ScorecardJobTypeVar,
)
from qai_hub_models.utils.path_helpers import get_git_branch


@dataclass
class ResultsSpreadsheet(list):
    has_compile_jobs: bool = True
    has_profile_jobs: bool = True
    has_inference_jobs: bool = True
    _model_metadata: dict[str, ResultsSpreadsheet.ModelMetadata] = field(
        default_factory=dict
    )
    _datestr: str | None = None
    _branchstr: str | None = None

    @dataclass
    class ModelMetadata:
        domain: MODEL_DOMAIN
        use_case: MODEL_USE_CASE
        tags: list[MODEL_TAG]
        known_failure_reasons: dict[TargetRuntime, str | None]

    @dataclass
    class Entry:
        model_id: str
        component_id: str | None
        precision: Precision
        chipset: str
        runtime: TargetRuntime
        quantize_status: str
        quantize_url: str | None
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
        # Default datetime if not set
        date: str
        if not self._datestr:
            self.set_date(datetime.now())
            date = cast(str, self._datestr)
            self._datestr = None
        else:
            date = self._datestr

        # Default to branch if not set
        branch = self._branchstr or get_git_branch()

        # Get CSV fields
        field_names = [field.name for field in fields(ResultsSpreadsheet.Entry)]

        # Add metadata fields
        field_names.insert(2, "domain")
        field_names.insert(3, "use_case")
        field_names.insert(4, "tags")
        field_names.insert(5, "branch")
        field_names.insert(6, "known_issue")

        # Add date field
        field_names.insert(2, "date")

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
        for field_name in fields_to_remove:
            field_names.remove(field_name)

        # Save CSV
        with open(path, "w") as csvfile:
            scorecard_csv = csv.writer(csvfile)

            # Header
            scorecard_csv.writerow(field_names)

            def _get_value(field_name: str, entry: ResultsSpreadsheet.Entry) -> str:
                model_id: str = entry.model_id
                if field_name == "model_id":
                    return (
                        entry.model_component_id
                        if combine_model_and_component_id
                        else entry.model_id
                    )
                if field_name == "date":
                    return date
                if field_name == "domain":
                    return (
                        self._model_metadata[model_id].domain.name
                        if model_id in self._model_metadata
                        else ""
                    )
                if field_name == "use_case":
                    return (
                        self._model_metadata[model_id].use_case.name
                        if model_id in self._model_metadata
                        else ""
                    )
                if field_name == "tags":
                    return (
                        ", ".join([x.name for x in self._model_metadata[model_id].tags])
                        if model_id in self._model_metadata
                        else ""
                    )
                if field_name == "branch":
                    return branch
                if field_name == "known_issue":
                    if meta := self._model_metadata.get(model_id):
                        if rt_reason := meta.known_failure_reasons.get(entry.runtime):
                            return rt_reason
                    return ""

                val = getattr(entry, field_name)
                if isinstance(val, bool):
                    return "YES" if val else "NO"
                return str(val)

            # Rows
            for entry in self:
                scorecard_csv.writerow(
                    [_get_value(name, entry) for name in field_names]
                )

    @dataclass
    class ModelInfo:
        model_id: str
        components: list[str] | None
        quantized: bool

    def append_model_summary_entries(
        self,
        precisions: list[Precision],
        quantize_summary: ModelQuantizeSummary | None = None,
        compile_summary: ModelCompileSummary | None = None,
        profile_summary: ModelPerfSummary | None = None,
        inference_summary: ModelInferenceSummary | None = None,
    ):
        self.extend(
            ResultsSpreadsheet.get_model_summary_entries(
                precisions,
                quantize_summary,
                compile_summary,
                profile_summary,
                inference_summary,
            )
        )

    def set_model_metadata(
        self,
        model_id: str,
        domain: MODEL_DOMAIN,
        use_case: MODEL_USE_CASE,
        tags: list[MODEL_TAG],
        known_failure_reasons: dict[TargetRuntime, str | None],
    ):
        self._model_metadata[model_id] = ResultsSpreadsheet.ModelMetadata(
            domain, use_case, tags, known_failure_reasons
        )

    def set_date(self, date: datetime | None):
        self._datestr = date.strftime("%m/%d/%Y") if date else None

    def set_branch(self, branch: str | None):
        self._branchstr = branch

    @staticmethod
    def get_model_summary_entries(
        precisions: list[Precision],
        quantize_summary: ModelQuantizeSummary | None = None,
        compile_summary: ModelCompileSummary | None = None,
        profile_summary: ModelPerfSummary | None = None,
        inference_summary: ModelInferenceSummary | None = None,
    ) -> list[ResultsSpreadsheet.Entry]:
        entries: list[ResultsSpreadsheet.Entry] = []

        # Validate input summaries match
        base_summary = (
            quantize_summary or compile_summary or profile_summary or inference_summary
        )
        if not base_summary:
            return []
        for summary in [
            quantize_summary,
            compile_summary,
            profile_summary,
            inference_summary,
        ]:
            for precision in precisions:
                if summary and (
                    precision not in base_summary.summaries_per_precision
                    or precision not in summary.summaries_per_precision
                    or not base_summary.summaries_per_precision[
                        precision
                    ].is_same_model(summary.summaries_per_precision[precision])
                ):
                    raise ValueError(
                        "Summaries do not point to the same model definition."
                    )

        def create_entry(
            precision: Precision, path: ScorecardProfilePath, device: ScorecardDevice
        ):
            nonlocal base_summary
            assert base_summary
            base_precision_summary = base_summary.summaries_per_precision[precision]
            nonlocal quantize_summary
            quantize_precision_summary = (
                quantize_summary.summaries_per_precision[precision]
                if quantize_summary
                else None
            )
            nonlocal compile_summary
            compile_precision_summary = (
                compile_summary.summaries_per_precision[precision]
                if compile_summary
                else None
            )
            nonlocal profile_summary
            profile_precision_summary = (
                profile_summary.summaries_per_precision[precision]
                if profile_summary
                else None
            )
            nonlocal inference_summary
            inference_precision_summary = (
                inference_summary.summaries_per_precision[precision]
                if inference_summary
                else None
            )

            # Extract model and component ids
            model_id = base_precision_summary.model_id
            component_ids = base_precision_summary.component_ids

            # Create empty summaries if a relevant summary is not passed.
            # Empty summaries will always return "skipped" jobs when queried for runs.
            quantize_precision_summary = (
                quantize_precision_summary
                or ModelPrecisionQuantizeSummary(
                    model_id, precision, None, {x: {} for x in component_ids}
                )
            )
            compile_precision_summary = (
                compile_precision_summary
                or ModelPrecisionCompileSummary(
                    model_id, precision, None, {x: {} for x in component_ids}
                )
            )
            profile_precision_summary = (
                profile_precision_summary
                or ModelPrecisionPerfSummary(
                    model_id, precision, None, {x: {} for x in component_ids}
                )
            )
            inference_precision_summary = (
                inference_precision_summary
                or ModelPrecisionInferenceSummary(
                    model_id, precision, None, {x: {} for x in component_ids}
                )
            )

            for component_id in component_ids:
                # Get job for this path + device + component combo
                quantize_job = quantize_precision_summary.get_run(
                    device, None, component_id
                )
                compile_job = compile_precision_summary.get_run(
                    device, path.compile_path, component_id
                )
                profile_job = profile_precision_summary.get_run(
                    device, path, component_id
                )
                inference_job = inference_precision_summary.get_run(
                    device, path, component_id
                )

                def _get_url_and_status(
                    sjob: ScorecardJobTypeVar,
                ) -> tuple[str, str | None]:
                    return (
                        sjob.job_status
                        + (f" ({sjob.status_message})" if sjob.status_message else ""),
                        sjob.job.url if not sjob.skipped else None,
                    )

                # Job status
                quantize_status, quantize_url = _get_url_and_status(quantize_job)
                compile_status, compile_url = _get_url_and_status(compile_job)
                profile_status, profile_url = _get_url_and_status(profile_job)
                inference_status, inference_url = _get_url_and_status(inference_job)

                # Profile job results
                if profile_job.success:
                    inference_time = float(profile_job.inference_time) / 1000
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
                    precision=precision,
                    chipset=device.chipset,
                    runtime=path.runtime,
                    quantize_status=quantize_status,
                    quantize_url=quantize_url,
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
            ScorecardProfilePath,
            create_entry,
            precisions,
            exclude_devices=[cs_universal],
            include_mirror_devices=True,
        )

        return entries

    def combine(self, other: ResultsSpreadsheet) -> None:
        self.extend(other)
        self._model_metadata.update(other._model_metadata)
