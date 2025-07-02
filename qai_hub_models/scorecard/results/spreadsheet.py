# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import cast

from qai_hub_models.configs.info_yaml import MODEL_DOMAIN, MODEL_TAG, MODEL_USE_CASE
from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import (
    for_each_scorecard_path_and_device,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_summary import (
    ModelCompileSummary,
    ModelInferenceSummary,
    ModelPerfSummary,
    ModelQuantizeSummary,
    ScorecardJobTypeVar,
)
from qai_hub_models.utils.path_helpers import get_git_branch

MAX_ERROR_LENGTH = 250
DEFAULT_MODEL_SUMMARY_ID = "__DEFAULT_SUMMARY_FOR_MODEL"


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
        known_failure_reasons: ModelDisableReasonsMapping
        public_status: str
        is_pytorch: bool

    @dataclass
    class Entry:
        model_id: str
        component_id: str | None
        precision: Precision
        chipset: str
        runtime: ScorecardProfilePath
        quantize_status: str
        quantize_url: str | None
        compile_status: str
        compile_url: str | None
        profile_status: str
        profile_url: str | None
        inference_time: float | None
        first_load_time: float | None
        warm_load_time: float | None
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
                if field_name == "runtime":
                    return entry.runtime.spreadsheet_name
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
                    if model_id not in self._model_metadata:
                        return ""
                    metadata = self._model_metadata[model_id]
                    tags = [x.name for x in metadata.tags]
                    tags.append(metadata.public_status)
                    tags.append("pytorch" if metadata.is_pytorch else "static")
                    return ", ".join(tags)
                if field_name == "branch":
                    return branch
                if field_name == "known_issue":
                    if meta := self._model_metadata.get(model_id):
                        if failure_reasons := meta.known_failure_reasons.get_disable_reasons(
                            entry.precision, entry.runtime.runtime
                        ):
                            if failure_reasons.has_failure:
                                return failure_reasons.failure_reason
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
        model_id: str,
        precisions: list[Precision],
        components: list[str] | None = None,
        devices: list[ScorecardDevice] | None = None,
        paths: list[ScorecardProfilePath] | None = None,
        quantize_summary: ModelQuantizeSummary | None = None,
        compile_summary: ModelCompileSummary | None = None,
        profile_summary: ModelPerfSummary | None = None,
        inference_summary: ModelInferenceSummary | None = None,
    ):
        self.extend(
            ResultsSpreadsheet.get_model_summary_entries(
                model_id,
                precisions,
                components,
                devices,
                paths,
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
        public_status: str,
        is_pytorch: bool,
        known_failure_reasons: ModelDisableReasonsMapping = ModelDisableReasonsMapping(),
    ):
        self._model_metadata[model_id] = ResultsSpreadsheet.ModelMetadata(
            domain, use_case, tags, known_failure_reasons, public_status, is_pytorch
        )

    def set_date(self, date: datetime | None):
        self._datestr = date.strftime("%m/%d/%Y") if date else None

    def set_branch(self, branch: str | None):
        self._branchstr = branch

    @staticmethod
    def get_model_summary_entries(
        model_id: str,
        precisions: list[Precision],
        components: list[str] | None = None,
        devices: list[ScorecardDevice] | None = None,
        paths: list[ScorecardProfilePath] | None = None,
        quantize_summary: ModelQuantizeSummary | None = None,
        compile_summary: ModelCompileSummary | None = None,
        profile_summary: ModelPerfSummary | None = None,
        inference_summary: ModelInferenceSummary | None = None,
    ) -> list[ResultsSpreadsheet.Entry]:
        entries: list[ResultsSpreadsheet.Entry] = []
        quantize_summary = quantize_summary or ModelQuantizeSummary()
        compile_summary = compile_summary or ModelCompileSummary()
        profile_summary = profile_summary or ModelPerfSummary()
        inference_summary = inference_summary or ModelInferenceSummary()

        def create_entry(
            precision: Precision, path: ScorecardProfilePath, device: ScorecardDevice
        ):
            for component_id in components or [model_id]:
                # Get job for this path + device + component combo
                quantize_job = quantize_summary.get_run(
                    precision, device, None, component_id
                )
                compile_job = compile_summary.get_run(
                    precision, device, path.compile_path, component_id
                )
                profile_job = profile_summary.get_run(
                    precision, device, path, component_id
                )
                inference_job = inference_summary.get_run(
                    precision, device, path, component_id
                )

                def _get_url_and_status(
                    sjob: ScorecardJobTypeVar,
                ) -> tuple[str, str | None]:
                    # Replace all whitespace with space character
                    status = (
                        re.sub(r"\s+", " ", sjob.status_message[:MAX_ERROR_LENGTH])
                        if sjob.status_message
                        else None
                    )
                    return (
                        sjob.job_status
                        + (f" ({status})" if sjob.status_message else ""),
                        sjob.job.url if not sjob.skipped else None,
                    )

                # Job status
                quantize_status, quantize_url = _get_url_and_status(quantize_job)
                compile_status, compile_url = _get_url_and_status(compile_job)
                profile_status, profile_url = _get_url_and_status(profile_job)
                inference_status, inference_url = _get_url_and_status(inference_job)

                # Profile job results
                if profile_job.success:
                    inference_time = profile_job.inference_time_milliseconds
                    first_load_time = profile_job.first_load_time_milliseconds
                    warm_load_time = profile_job.warm_load_time_milliseconds
                    NPU = profile_job.layer_counts.npu
                    GPU = profile_job.layer_counts.gpu
                    CPU = profile_job.layer_counts.cpu
                else:
                    inference_time = None
                    first_load_time = None
                    warm_load_time = None
                    NPU = None
                    GPU = None
                    CPU = None

                # Create Entry
                entry = ResultsSpreadsheet.Entry(
                    model_id=model_id,
                    component_id=component_id if component_id != model_id else None,
                    precision=precision,
                    chipset=device.chipset,
                    runtime=path,
                    quantize_status=quantize_status,
                    quantize_url=quantize_url,
                    compile_status=compile_status.replace(
                        ",", "."
                    ),  # remove commas for CSV compatibliity
                    compile_url=compile_url,
                    profile_status=profile_status.replace(
                        ",", "."
                    ),  # remove commas for CSV compatibliity
                    profile_url=profile_url,
                    inference_time=inference_time,
                    first_load_time=first_load_time,
                    warm_load_time=warm_load_time,
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
            include_devices=devices,
            include_paths=paths,
        )

        return entries

    def combine(self, other: ResultsSpreadsheet) -> None:
        self.extend(other)
        self._model_metadata.update(other._model_metadata)
