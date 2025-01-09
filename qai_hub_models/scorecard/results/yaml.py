# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Generic, Optional, TypeVar, cast

import ruamel.yaml

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import get_async_job_cache_name
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_summary import (
    CompileSummary,
    InferenceSummary,
    PerfSummary,
    ScorecardSummaryTypeVar,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    ProfileScorecardJob,
    ScorecardJobTypeVar,
    ScorecardPathTypeVar,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_PACKAGE_ROOT

INTERMEDIATES_DIR = QAIHM_PACKAGE_ROOT / "scorecard" / "intermediates"
COMPILE_YAML_BASE = INTERMEDIATES_DIR / "compile-jobs.yaml"
PROFILE_YAML_BASE = INTERMEDIATES_DIR / "profile-jobs.yaml"
ScorecardJobYamlTypeVar = TypeVar("ScorecardJobYamlTypeVar", bound="ScorecardJobYaml")


class ScorecardJobYaml(
    Generic[ScorecardJobTypeVar, ScorecardPathTypeVar, ScorecardSummaryTypeVar]
):
    scorecard_job_type: type[ScorecardJobTypeVar]
    scorecard_path_type: type[ScorecardPathTypeVar]
    scorecard_summary_type: type[ScorecardSummaryTypeVar]

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

    def to_file(self, path: str | Path) -> None:
        yaml = ruamel.yaml.YAML()
        with open(path, "w") as file:
            yaml.dump(self.job_id_mapping, file)

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
                device,
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
    ) -> ScorecardJobTypeVar:
        """
        Get the scorecard job from the YAML associated with these parameters.

        parameters:
            path: Applicable scorecard path
            model_id: The ID of the QAIHM model being tested
            device: The targeted device
            component: The name of the model component being tested, if applicable
        """
        return self.scorecard_job_type(
            component or model_id,
            self.get_job_id(
                path, model_id, device, component, fallback_to_universal_device=True
            ),
            device,
            True,
            None,
            path,  # type: ignore
        )

    def get_jobs_from_model_info(
        self, model_info: QAIHMModelInfo
    ) -> list[ScorecardJobTypeVar]:
        """
        Get all jobs in this YAML related to the model information in the given model info class.
        """
        components: list[Optional[str]] = []
        if model_info.code_gen_config.components:
            if model_info.code_gen_config.default_components:
                components = cast(
                    list[Optional[str]], model_info.code_gen_config.default_components
                )
            else:
                components = list(model_info.code_gen_config.components.keys())
        else:
            components.append(None)

        supports_fp16_npu = not (
            model_info.code_gen_config.is_aimet
            or model_info.code_gen_config.use_hub_quantization
        )

        model_runs = []
        for path in self.scorecard_path_type.all_paths(enabled=True):
            for component in components:
                for device in ScorecardDevice.all_devices(
                    enabled=True,
                    supports_fp16_npu=supports_fp16_npu or None,
                    supports_compile_path=path
                    if isinstance(path, ScorecardCompilePath)
                    else None,
                    supports_profile_path=path
                    if isinstance(path, ScorecardProfilePath)
                    else None,
                ):
                    job = self.get_job(
                        path, model_info.id, device, component  # type: ignore
                    )
                    if not component:
                        job.model_id = model_info.name

                    model_runs.append(job)

        return model_runs

    def summaries_from_model_ids(
        self, model_ids: list[str] = MODEL_IDS
    ) -> dict[str, ScorecardSummaryTypeVar]:
        """
        Create a summary for each set of jobs related to each model id in the provided list.

        Returns models in this format:
            model_id: list[Summary]
        """
        print(f"Generating {self.scorecard_summary_type.__name__} for Models")
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.map(
            self.summary_from_model_id,
            model_ids,
        )
        pool.close()
        print("Finished\n")
        return {k: v for k, v in zip(model_ids, model_summaries)}

    def summary_from_model_id(self, model_id: str) -> ScorecardSummaryTypeVar:
        """
        Creates a summary of all jobs related to the given model id.
        """
        print(f"    {model_id} ")
        runs = self.get_jobs_from_model_info(QAIHMModelInfo.from_model(model_id))
        return self.scorecard_summary_type.from_runs(runs)  # type: ignore


class CompileScorecardJobYaml(
    ScorecardJobYaml[CompileScorecardJob, ScorecardCompilePath, CompileSummary]
):
    scorecard_job_type = CompileScorecardJob
    scorecard_path_type = ScorecardCompilePath
    scorecard_summary_type = CompileSummary


class ProfileScorecardJobYaml(
    ScorecardJobYaml[ProfileScorecardJob, ScorecardProfilePath, PerfSummary]
):
    scorecard_job_type = ProfileScorecardJob
    scorecard_path_type = ScorecardProfilePath
    scorecard_summary_type = PerfSummary


class InferenceScorecardJobYaml(
    ScorecardJobYaml[InferenceScorecardJob, ScorecardProfilePath, InferenceSummary]
):
    scorecard_job_type = InferenceScorecardJob
    scorecard_path_type = ScorecardProfilePath
    scorecard_summary_type = InferenceSummary
