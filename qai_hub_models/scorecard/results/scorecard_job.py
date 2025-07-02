# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import datetime
import time
from functools import cached_property
from typing import Any, Generic, Optional, TypeVar, cast

import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)

JobTypeVar = TypeVar(
    "JobTypeVar", hub.ProfileJob, hub.InferenceJob, hub.CompileJob, hub.QuantizeJob
)
ScorecardPathTypeVar = TypeVar(
    "ScorecardPathTypeVar", ScorecardCompilePath, ScorecardProfilePath
)
ScorecardPathOrNoneTypeVar = TypeVar(
    "ScorecardPathOrNoneTypeVar", ScorecardCompilePath, ScorecardProfilePath, None
)

# Specific typevar. Autofill has trouble resolving types for nested generics without specifically listing inheritors of the generic base.
ScorecardJobTypeVar = TypeVar(
    "ScorecardJobTypeVar",
    "QuantizeScorecardJob",
    "CompileScorecardJob",
    "ProfileScorecardJob",
    "InferenceScorecardJob",
)


class ScorecardJob(Generic[JobTypeVar, ScorecardPathOrNoneTypeVar]):
    job_type_class: type[JobTypeVar]

    def __init__(
        self,
        model_id: str,
        precision: Precision,
        job_id: Optional[str],
        device: ScorecardDevice,
        wait_for_job: bool,  # If false, running jobs are treated like they were "skipped".
        wait_for_max_job_duration: Optional[
            int
        ],  # Allow the job this many seconds after creation to complete
        path: ScorecardPathOrNoneTypeVar,
    ):
        self.model_id = model_id
        self.precision = precision
        self.job_id = job_id
        self._device = device
        self.wait_for_job = wait_for_job
        self.wait_for_max_job_duration = wait_for_max_job_duration
        self.path: ScorecardPathOrNoneTypeVar = path
        self.__post_init__()

    def __post_init__(self):
        assert self.model_id
        # Verify Job Exists
        if self.job_id and not self.wait_for_job:
            assert self.job

        if not self.skipped and not isinstance(self.job, self.job_type_class):
            raise ValueError(
                f"Job {self.job.job_id}({self.job.name}) is {type(self.job)}. Expected {self.job_type_class.__name__}"
            )

    @cached_property
    def job(self) -> JobTypeVar:
        """
        Get the AI Hub Job.
        Waits for completion if necessary.
        """
        if not self.job_id:
            raise ValueError("No Job ID")

        job = cast(JobTypeVar, hub.get_job(self.job_id))
        if not job.get_status().finished:
            if not self.wait_for_job:
                return job
            else:
                if self.wait_for_max_job_duration:
                    time_left = int(
                        job.date.timestamp()
                        + self.wait_for_max_job_duration
                        - time.time()
                    )
                    job.wait(time_left)
                else:
                    job.wait()
        return job

    @cached_property
    def skipped(self) -> bool:
        #
        # Running is treated as skipped.
        #
        # Either the class would have waited for this job already,
        # or the class was told to treat running jobs like they were skipped.
        #
        return not self.job_id or self._job_status.running

    @cached_property
    def failed(self) -> bool:
        return not self.skipped and self._job_status.failure

    @cached_property
    def success(self) -> bool:
        return not self.skipped and self._job_status.success

    @cached_property
    def status_message(self) -> Optional[str]:
        return None if self.skipped else self._job_status.message

    @cached_property
    def _job_status(self) -> hub.JobStatus:
        """Get the job status of the profile job."""
        if self.job_id:
            return self.job.get_status()
        raise ValueError("Can't get status without a job ID.")

    @cached_property
    def job_status(self) -> str:
        """Get the job status of the profile job."""
        if not self.skipped:
            if self._job_status.success:
                return "Passed"
            elif self._job_status.failure:
                return "Failed"
        return "Skipped"

    @cached_property
    def device(self) -> hub.Device:
        if not self.skipped and not isinstance(self.job, hub.QuantizeJob):
            return self.job.device
        return self._device.reference_device

    @cached_property
    def chipset(self) -> str:
        """Chipset the job was run on."""
        if self.skipped:
            return self._device.chipset

        hub_device = self.device
        for attr in hub_device.attributes:
            if attr.startswith("chipset:"):
                return attr.split(":")[1]
        raise ValueError("No chipset found.")

    @cached_property
    def date(self) -> Optional[datetime.datetime]:
        if self.job is None:
            return None
        return self.job.date


class QuantizeScorecardJob(ScorecardJob[hub.QuantizeJob, ScorecardCompilePath]):
    job_type_class = hub.QuantizeJob


class CompileScorecardJob(ScorecardJob[hub.CompileJob, ScorecardCompilePath]):
    job_type_class = hub.CompileJob


class ProfileScorecardJob(ScorecardJob[hub.ProfileJob, ScorecardProfilePath]):
    job_type_class = hub.ProfileJob

    def __post_init__(self):
        super().__post_init__()
        if not self.skipped and self._job_status.success:
            assert self.profile_results  # Download results immediately

    @cached_property
    def profile_results(self) -> dict[str, Any]:
        """Profile results from profile job."""
        if self.success:
            profile = self.job.download_profile()
            assert isinstance(profile, dict)
            return profile
        raise ValueError("Can't get profile results if job did not succeed.")

    @cached_property
    def inference_time_milliseconds(self) -> float:
        """Get the inference time from the profile job."""
        return float(
            self.profile_results["execution_summary"]["estimated_inference_time"] / 1000
        )

    @cached_property
    def first_load_time_milliseconds(self) -> float:
        """Get the first load time from the profile job."""
        return float(
            self.profile_results["execution_summary"]["first_load_time"] / 1000
        )

    @cached_property
    def warm_load_time_milliseconds(self) -> float:
        """Get the warm load time from the profile job."""
        return float(self.profile_results["execution_summary"]["warm_load_time"] / 1000)

    @cached_property
    def layer_counts(self) -> QAIHMModelPerf.PerformanceDetails.LayerCounts:
        """Count layers per compute unit."""

        def _count_unit(unit: str) -> int:
            return sum(
                1
                for detail in self.profile_results["execution_detail"]
                if detail["compute_unit"] == unit
            )

        cpu = _count_unit("CPU")
        gpu = _count_unit("GPU")
        npu = _count_unit("NPU")

        return QAIHMModelPerf.PerformanceDetails.LayerCounts.from_layers(npu, gpu, cpu)

    @cached_property
    def estimated_peak_memory_range_mb(
        self,
    ) -> QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB:
        """Get the estimated peak memory range."""
        low, high = self.profile_results["execution_summary"][
            "inference_memory_peak_range"
        ]
        return QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB.from_bytes(low, high)

    @cached_property
    def performance_metrics(self) -> QAIHMModelPerf.PerformanceDetails:
        metrics = QAIHMModelPerf.PerformanceDetails(
            job_id=self.job_id,
            job_status=self.job_status,
            inference_time_milliseconds=(
                self.inference_time_milliseconds if self.success else None
            ),
            estimated_peak_memory_range_mb=(
                self.estimated_peak_memory_range_mb if self.success else None
            ),
            primary_compute_unit=(
                self.layer_counts.primary_compute_unit if self.success else None
            ),
            layer_counts=self.layer_counts if self.success else None,
        )
        return metrics


class InferenceScorecardJob(ScorecardJob[hub.InferenceJob, ScorecardProfilePath]):
    job_type_class = hub.InferenceJob

    @property
    def input_dataset(self) -> DatasetEntries:
        """Input dataset."""
        return cast(DatasetEntries, self.job.inputs.download())

    @property
    def output_dataset(self) -> DatasetEntries:
        """Output dataset."""
        if not self.success:
            raise ValueError("Can't get output dataset if job did not succeed.")
        return cast(DatasetEntries, self.job.download_output_data())
