# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import datetime
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Type, Union, cast

import qai_hub as hub

from qai_hub_models.utils.config_loaders import QAIHMModelCodeGen, QAIHMModelInfo
from qai_hub_models.utils.scorecard.common import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)


@dataclass
class JobSummary:
    model_id: str
    job_id: Optional[str]
    _device: ScorecardDevice
    # Setting for how the JobSummary class should treat a job.
    #  None | Wait an infinite amount of time the job to finish
    #   < 0 | Ignore job if running (treat it as skipped)
    #  >= 0 | Wait this many seconds for the job to finish
    max_job_wait_secs: Optional[int]

    def __post_init__(self):
        assert self.model_id
        # Verify Job Exists
        if self.job_id and (not self.max_job_wait_secs or self.max_job_wait_secs >= 0):
            assert self.job

    @classmethod
    def from_model_id(
        cls: Type["JobSummary"], model_id: str, job_ids: Dict[str, str]
    ) -> List:
        """
        Reads jobs for `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>|<runtime>|<device>|<model_component_id> : job_id
            <model_id>|<runtime>|<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        raise NotImplementedError()

    @cached_property
    def job(self) -> Optional[hub.Job]:
        """Get the hub.CompileJob object."""
        if not self.job_id:
            return None

        job = hub.get_job(self.job_id)
        if job.get_status().running:
            if self.max_job_wait_secs and self.max_job_wait_secs < 0:
                return None
            else:
                job.wait(self.max_job_wait_secs)
        return job

    @cached_property
    def skipped(self) -> bool:
        return self.job is None

    @cached_property
    def failed(self) -> bool:
        return self._job_status and self._job_status.failure  # type: ignore

    @cached_property
    def success(self) -> bool:
        return self._job_status and self._job_status.success  # type: ignore

    @cached_property
    def status_message(self) -> str:
        return "Skipped" if self.skipped else self._job_status.message  # type: ignore

    @cached_property
    def _job_status(self) -> Optional[hub.JobStatus]:
        """Get the job status of the profile job."""
        if not self.skipped:
            return self.job.get_status()  # type: ignore
        return None

    @cached_property
    def job_status(self) -> str:
        """Get the job status of the profile job."""
        if not self.skipped:
            if self._job_status.success:  # type: ignore
                return "Passed"
            elif self._job_status.failure:  # type: ignore
                return "Failed"
        return "Skipped"

    @cached_property
    def quantized(self) -> str:
        """Quantized models are marked so precision can be correctly recorded."""
        return (
            "Yes"
            if self.model_id.endswith("Quantized")
            or self.model_id.endswith("Quantizable")
            else "No"
        )

    @cached_property
    def date(self) -> Optional[datetime.datetime]:
        if self.job is None:
            return None
        return self.job.date


@dataclass
class CompileJobSummary(JobSummary):
    path: ScorecardCompilePath

    @classmethod
    def from_model_id(
        cls: Type["CompileJobSummary"],
        model_id: str,
        job_ids: Dict[str, str],
        max_job_wait_secs=None,
    ) -> List["CompileJobSummary"]:
        """
        Reads jobs for `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>|<runtime>|<device>|<model_component_id> : job_id
            <model_id>|<runtime>|<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        model_info = QAIHMModelInfo.from_model(model_id)
        model_code_gen: QAIHMModelCodeGen = model_info.code_gen_config
        model_runs = []
        components = []

        if model_code_gen.components:
            if model_code_gen.default_components:
                components = model_code_gen.default_components
            else:
                components = list(model_code_gen.components.keys())
        else:
            components.append(None)  # type: ignore

        path: ScorecardCompilePath
        for path in ScorecardCompilePath.all_enabled():
            for component in components:
                path_devices_enabled = [
                    x
                    for x in path.get_test_devices(model_code_gen.is_aimet)
                    if x.enabled()
                ]
                for device in path_devices_enabled:
                    model_runs.append(
                        cls(
                            model_id=component or model_info.name,
                            job_id=job_ids.get(
                                path.get_job_cache_name(
                                    model=model_id,
                                    device=device,
                                    component=component,
                                )
                            ),
                            path=path,
                            _device=device,
                            max_job_wait_secs=max_job_wait_secs,
                        )
                    )

        return model_runs

    def __post_init__(self):
        super().__post_init__()
        if not self.skipped:
            assert isinstance(self.job, hub.CompileJob)

    @cached_property
    def compile_job(self) -> Optional[hub.CompileJob]:
        """Get the hub.CompileJob object."""
        if self.job:
            return None
        return cast(hub.CompileJob, self.job)


@dataclass
class ProfileJobSummary(JobSummary):
    path: ScorecardProfilePath

    @classmethod
    def from_model_id(
        cls: Type["ProfileJobSummary"],
        model_id: str,
        job_ids: Dict[str, str],
        max_job_wait_secs=None,
    ) -> List["ProfileJobSummary"]:
        """
        Reads jobs for `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>|<runtime>|<device>|<model_component_id> : job_id
            <model_id>|<runtime>|<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        model_info = QAIHMModelInfo.from_model(model_id)
        model_code_gen: QAIHMModelCodeGen = model_info.code_gen_config
        model_runs = []
        components = []

        if model_code_gen.components:
            if model_code_gen.default_components:
                components = model_code_gen.default_components
            else:
                components = list(model_code_gen.components.keys())
        else:
            components.append(None)  # type: ignore

        path: ScorecardProfilePath
        for path in ScorecardProfilePath.all_enabled():
            for component in components:
                path_devices_enabled = [
                    x
                    for x in path.get_test_devices(model_code_gen.is_aimet)
                    if x.enabled()
                ]
                for device in path_devices_enabled:
                    model_runs.append(
                        cls(
                            model_id=component or model_info.name,
                            job_id=job_ids.get(
                                path.get_job_cache_name(
                                    model=model_id,
                                    device=device,
                                    component=component,
                                ),
                                None,
                            ),
                            _device=device,
                            path=path,
                            max_job_wait_secs=max_job_wait_secs,
                        )
                    )

        return model_runs

    def __post_init__(self):
        super().__post_init__()
        if not self.skipped:
            assert isinstance(self.job, hub.ProfileJob)
            if self._job_status.success:
                assert self.profile_results

    @cached_property
    def chipset(self) -> str:
        """Chipset the job was run on."""
        if not self.job:
            return self._device.get_chipset()

        hub_device = self.job.device
        for attr in hub_device.attributes:
            if attr.startswith("chipset:"):
                return attr.split(":")[1]
        raise ValueError("No chipset found.")

    @cached_property
    def device(self) -> hub.Device:
        return self.job.device if self.job else self._device.get_reference_device()

    @cached_property
    def profile_job(self) -> Optional[hub.ProfileJob]:
        """Get the hub.CompileJob object."""
        if not self.job:
            return None
        return cast(hub.ProfileJob, self.job)

    @cached_property
    def profile_results(self) -> Optional[Dict[str, Any]]:
        """Profile results from profile job."""
        if self.job_status == "Passed":
            return self.profile_job.download_profile()  # type: ignore
        return None

    @cached_property
    def inference_time(self) -> Union[float, str]:
        """Get the inference time from the profile job."""
        if self.profile_results is not None:
            return float(
                self.profile_results["execution_summary"]["estimated_inference_time"]
            )
        return "null"

    @cached_property
    def throughput(self) -> Union[float, str]:
        """Get the throughput from the profile job."""
        if not isinstance(self.inference_time, str):
            return 1000000 / self.inference_time  # type: ignore
        return "null"

    def get_layer_info(self, unit: str) -> int:
        """Count layers per compute unit."""
        if self.profile_results is not None:
            count: int = 0
            count = sum(
                1
                for detail in self.profile_results["execution_detail"]
                if detail["compute_unit"] == unit
            )
            return count
        return 0

    @cached_property
    def npu(self) -> Any:
        """Get number of layers running on NPU."""
        return self.get_layer_info("NPU") if self.profile_results is not None else 0

    @cached_property
    def gpu(self) -> Any:
        """Get number of layers running on GPU."""
        return self.get_layer_info("GPU") if self.profile_results is not None else 0

    @cached_property
    def cpu(self) -> Any:
        """Get number of layers running on CPU."""
        return self.get_layer_info("CPU") if self.profile_results is not None else 0

    @cached_property
    def total(self) -> Any:
        """Get the total number of layers."""
        return self.npu + self.gpu + self.cpu

    @cached_property
    def primary_compute_unit(self) -> str:
        """Get the primary compute unit."""
        layers_npu = self.npu
        layers_gpu = self.gpu
        layers_cpu = self.cpu

        if layers_npu == 0 and layers_gpu == 0 and layers_cpu == 0:
            return "null"
        compute_unit_for_most_layers = max(layers_cpu, layers_gpu, layers_npu)
        if compute_unit_for_most_layers == layers_npu:
            return "NPU"
        elif compute_unit_for_most_layers == layers_gpu:
            return "GPU"
        return "CPU"

    @cached_property
    def peak_memory_range(self) -> Dict[str, int]:
        """Get the estimated peak memory range."""
        if self.profile_results is not None:
            low, high = self.profile_results["execution_summary"][
                "inference_memory_peak_range"
            ]
            return dict(min=low, max=high)
        return dict(min=0, max=0)

    @cached_property
    def precision(self) -> str:
        """Get the precision of the model based on the run."""
        if self.profile_results is not None:
            compute_unit = self.primary_compute_unit
            if compute_unit == "CPU":
                return "fp32"
            if self.quantized == "Yes":
                return "int8"
            return "fp16"
        return "null"

    @cached_property
    def performance_metrics(self) -> Dict[str, Any]:
        return dict(
            inference_time=self.inference_time,
            throughput=self.throughput,
            estimated_peak_memory_range=self.peak_memory_range,
            primary_compute_unit=self.primary_compute_unit,
            precision=self.precision,
            layer_info=dict(
                layers_on_npu=self.npu,
                layers_on_gpu=self.gpu,
                layers_on_cpu=self.cpu,
                total_layers=self.total,
            ),
            job_id=self.job_id,
            job_status=self.job_status,
        )
