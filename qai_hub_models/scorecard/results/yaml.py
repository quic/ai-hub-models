# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, Literal, Optional, TypeVar, overload

import qai_hub as hub
import ruamel.yaml

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import (
    for_each_scorecard_path_and_device,
    get_async_job_cache_name,
)
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_summary import (
    ModelCompileSummary,
    ModelInferenceSummary,
    ModelPerfSummary,
    ModelQuantizeSummary,
    ModelSummaryTypeVar,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    ProfileScorecardJob,
    QuantizeScorecardJob,
    ScorecardJobTypeVar,
    ScorecardPathTypeVar,
)
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

INTERMEDIATES_DIR = QAIHM_PACKAGE_ROOT / "scorecard" / "intermediates"

QUANTIZE_YAML_BASE = INTERMEDIATES_DIR / "quantize-jobs.yaml"
COMPILE_YAML_BASE = INTERMEDIATES_DIR / "compile-jobs.yaml"
PROFILE_YAML_BASE = INTERMEDIATES_DIR / "profile-jobs.yaml"
INFERENCE_YAML_BASE = INTERMEDIATES_DIR / "inference-jobs.yaml"
DATASETS_BASE = INTERMEDIATES_DIR / "dataset-ids.yaml"
ScorecardJobYamlTypeVar = TypeVar("ScorecardJobYamlTypeVar", bound="ScorecardJobYaml")


class ScorecardJobYaml(
    Generic[ScorecardJobTypeVar, ScorecardPathTypeVar, ModelSummaryTypeVar]
):
    scorecard_job_type: type[ScorecardJobTypeVar]
    scorecard_path_type: type[ScorecardPathTypeVar]
    scorecard_model_summary_type: type[ModelSummaryTypeVar]

    def __init__(self, job_id_mapping: dict[str, str] | None = None):
        self.job_id_mapping = job_id_mapping or dict()

    @classmethod
    def from_file(
        cls: type[ScorecardJobYamlTypeVar], config_path: str | Path
    ) -> ScorecardJobYamlTypeVar:
        """Read yaml files."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File not found with job ids at {config_path}")

        yaml = ruamel.yaml.YAML()
        with open(config_path) as file:
            return cls(yaml.load(file))

    def to_file(self, path: str | Path, append: bool = False) -> None:
        if len(self.job_id_mapping) > 0:
            with open(path, "a" if append else "w") as yaml_file:
                ruamel.yaml.YAML().dump(self.job_id_mapping, yaml_file)
        elif not append:
            # If the dict is empty, ruamel dumps "{}" (which is not YAML) and breaks the file
            Path(path).touch()

    def clear_jobs(self, model_id: str | None = None):
        if not model_id:
            self.job_id_mapping.clear()
        else:
            # find jobs to delete
            # catch "model", ignore "model_quantized"
            keys_to_delete = [
                key
                for key in self.job_id_mapping
                if (model_id in key and f"{model_id}_quantized" not in key)
            ]

            # Delete keys
            for key in keys_to_delete:
                del self.job_id_mapping[key]

    def get_job_id(
        self,
        path: ScorecardPathTypeVar | TargetRuntime,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
        fallback_to_universal_device: bool = False,
    ) -> str | None:
        """
        Get the ID of this job in the YAML that stores asyncronously-ran scorecard jobs.
        Returns None if job does not exist.

        parameters:
            path: Applicable scorecard path
            model_id: The ID of the QAIHM model being tested
            device: The targeted device
            component: The name of the model component being tested, if applicable
            fallback_to_universal_device: Return a job that ran with the universal device if a job
                                        using the provided device is not available.
        """
        if x := self.job_id_mapping.get(
            get_async_job_cache_name(
                path,
                model_id,
                device.mirror_device or device,
                component,
            )
        ):
            return x

        if fallback_to_universal_device:
            return self.job_id_mapping.get(
                get_async_job_cache_name(
                    path,
                    model_id,
                    cs_universal,
                    component,
                )
            )

        return None

    def set_job_id(
        self,
        job_id,
        path: ScorecardPathTypeVar | TargetRuntime,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ) -> None:
        """
        Set the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

        parameters:
            job_id: Job ID to associate with the other parameters in the YAML
            path: Applicable scorecard path
            model_id: The ID of the QAIHM model being tested
            device: The targeted device
            component: The name of the model component being tested, if applicable
        """
        self.job_id_mapping[
            get_async_job_cache_name(path, model_id, device, component)
        ] = job_id

    def update(self, other: ScorecardJobYaml):
        """
        Merge the other YAML into this YAML, overwriting any existing jobs with the same job name
        """
        if type(other) is not type(self):
            raise ValueError(
                f"Cannot merge scorecard YAMLS of types {type(other)} and {type(self)}"
            )
        self.job_id_mapping.update(other.job_id_mapping)

    def get_job(
        self,
        path: ScorecardPathTypeVar,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
        wait_for_job: bool = True,
        wait_for_max_job_duration: Optional[int] = None,
    ) -> ScorecardJobTypeVar:
        """
        Get the scorecard job from the YAML associated with these parameters.

        parameters:
            path: Applicable scorecard path
            model_id: The ID of the QAIHM model being tested
            device: The targeted device
            wait_for_job:  If false, running jobs are treated like they were "skipped"
            wait_job_secs: Wait a set number of seconds for a job to finish
            wait_for_max_job_duration: Allow the job this many seconds after creation to complete
            component: The name of the model component being tested, if applicable
        """
        return self.scorecard_job_type(
            component or model_id,
            self.get_job_id(
                path, model_id, device, component, fallback_to_universal_device=True
            ),
            device,
            wait_for_job,
            wait_for_max_job_duration,
            path,  # type: ignore[arg-type]
        )

    def get_all_jobs(
        self,
        model_id: str,
        is_quantized: bool,
        components: Iterable[str] | None = None,
    ) -> list[ScorecardJobTypeVar]:
        """
        Get all jobs in this YAML related to the provided model.
        """
        model_runs: list[ScorecardJobTypeVar] = []
        for component in components or [None]:  # type: ignore[list-item]

            def create_job(path: ScorecardPathTypeVar, device: ScorecardDevice):
                model_runs.append(
                    self.get_job(path, model_id, device, component or None)
                )

            for_each_scorecard_path_and_device(
                is_quantized,
                self.__class__.scorecard_path_type,
                create_job,
                include_mirror_devices=True,
            )

        return model_runs

    def summary_from_model(
        self,
        model_id: str,
        is_quantized: bool,
        components: Iterable[str] | None = None,
    ) -> ModelSummaryTypeVar:
        """
        Creates a summary of all jobs related to the given model.
        """
        runs = self.get_all_jobs(model_id, is_quantized, components)
        return self.scorecard_model_summary_type.from_runs(model_id, runs, components)  # type: ignore[arg-type]


class QuantizeScorecardJobYaml(
    ScorecardJobYaml[QuantizeScorecardJob, ScorecardCompilePath, ModelQuantizeSummary]
):
    scorecard_job_type = QuantizeScorecardJob
    scorecard_path_type = ScorecardCompilePath
    scorecard_model_summary_type = ModelQuantizeSummary

    def get_job_id(
        self,
        path: ScorecardPathTypeVar | TargetRuntime,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
        fallback_to_universal_device: bool = False,
    ) -> str | None:
        return self.job_id_mapping.get(
            get_async_job_cache_name(None, model_id, cs_universal, component)
        )

    def set_job_id(
        self,
        job_id,
        path: ScorecardPathTypeVar | TargetRuntime,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ) -> None:
        self.job_id_mapping[
            get_async_job_cache_name(None, model_id, cs_universal, component)
        ] = job_id

    def get_all_jobs(
        self,
        model_id: str,
        is_quantized: bool,
        components: Iterable[str] | None = None,
    ) -> list[QuantizeScorecardJob]:
        model_runs: list[QuantizeScorecardJob] = []
        for component in components or [None]:  # type: ignore

            def create_job(path: ScorecardCompilePath, device: ScorecardDevice):
                model_runs.append(
                    self.get_job(path, model_id, device, component or None)
                )

            for_each_scorecard_path_and_device(
                is_quantized,
                self.__class__.scorecard_path_type,
                create_job,
                include_paths=[ScorecardCompilePath.ONNX],
                include_devices=[cs_universal],
            )
        return model_runs


class CompileScorecardJobYaml(
    ScorecardJobYaml[CompileScorecardJob, ScorecardCompilePath, ModelCompileSummary]
):
    scorecard_job_type = CompileScorecardJob
    scorecard_path_type = ScorecardCompilePath
    scorecard_model_summary_type = ModelCompileSummary


class ProfileScorecardJobYaml(
    ScorecardJobYaml[ProfileScorecardJob, ScorecardProfilePath, ModelPerfSummary]
):
    scorecard_job_type = ProfileScorecardJob
    scorecard_path_type = ScorecardProfilePath
    scorecard_model_summary_type = ModelPerfSummary


class InferenceScorecardJobYaml(
    ScorecardJobYaml[InferenceScorecardJob, ScorecardProfilePath, ModelInferenceSummary]
):
    scorecard_job_type = InferenceScorecardJob
    scorecard_path_type = ScorecardProfilePath
    scorecard_model_summary_type = ModelInferenceSummary


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.COMPILE], path: str | Path | None = None
) -> CompileScorecardJobYaml:
    ...


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.PROFILE], path: str | Path | None = None
) -> ProfileScorecardJobYaml:
    ...


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.INFERENCE], path: str | Path | None = None
) -> InferenceScorecardJobYaml:
    ...


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.QUANTIZE], path: str | Path | None = None
) -> QuantizeScorecardJobYaml:
    ...


@overload
def get_scorecard_job_yaml(
    job_type: hub.JobType, path: str | Path | None = None
) -> ScorecardJobYaml:
    ...


def get_scorecard_job_yaml(
    job_type: hub.JobType, path: str | Path | None = None
) -> ScorecardJobYaml:
    """
    Loads the appropriate Scorecard job cache for the type of the given job.
    """
    if job_type == hub.JobType.COMPILE:
        return (
            CompileScorecardJobYaml()
            if not path
            else CompileScorecardJobYaml.from_file(path)
        )
    elif job_type == hub.JobType.PROFILE:
        return (
            ProfileScorecardJobYaml()
            if not path
            else ProfileScorecardJobYaml.from_file(path)
        )
    elif job_type == hub.JobType.INFERENCE:
        return (
            InferenceScorecardJobYaml()
            if not path
            else InferenceScorecardJobYaml.from_file(path)
        )
    elif job_type == hub.JobType.QUANTIZE:
        return (
            QuantizeScorecardJobYaml()
            if not path
            else QuantizeScorecardJobYaml.from_file(path)
        )
    else:
        raise NotImplementedError(
            f"No file for storing test jobs of type {job_type.display_name}"
        )


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.CompileJob, path: str | Path | None = None
) -> CompileScorecardJobYaml:
    ...


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.ProfileJob, path: str | Path | None = None
) -> ProfileScorecardJobYaml:
    ...


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.InferenceJob, path: str | Path | None = None
) -> InferenceScorecardJobYaml:
    ...


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.QuantizeJob, path: str | Path | None = None
) -> QuantizeScorecardJobYaml:
    ...


def get_scorecard_job_yaml_from_job(
    job: hub.Job, path: str | Path | None = None
) -> ScorecardJobYaml:
    return get_scorecard_job_yaml(job._job_type, path)
