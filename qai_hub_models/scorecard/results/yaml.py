# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Any, Optional

import ruamel.yaml

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import get_async_job_cache_name
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    ProfileScorecardJob,
)
from qai_hub_models.utils.path_helpers import get_qaihm_package_root

INTERMEDIATES_DIR = get_qaihm_package_root() / "scorecard" / "intermediates"
COMPILE_YAML_BASE = INTERMEDIATES_DIR / "compile-jobs.yaml"
PROFILE_YAML_BASE = INTERMEDIATES_DIR / "profile-jobs.yaml"


class ScorecardJobYaml:
    def __init__(self, job_id_mapping: dict[str, str] | None = None):
        self.job_id_mapping = job_id_mapping or dict()

    @classmethod
    def from_file(cls, config_path: str | Path) -> ScorecardJobYaml:
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
        path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime,
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
            get_async_job_cache_name(path, model_id, device, component)
        ):
            return x

        if fallback_to_universal_device:
            return self.job_id_mapping.get(
                get_async_job_cache_name(path, model_id, cs_universal, component)
            )

        return None

    def set_job_id(
        self,
        job_id,
        path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime,
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

    @abc.abstractmethod
    def get_job(
        self,
        path: Any,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ) -> CompileScorecardJob | ProfileScorecardJob:
        pass


class CompileScorecardJobYaml(ScorecardJobYaml):
    def get_job(
        self,
        path: ScorecardCompilePath,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ) -> CompileScorecardJob:
        """
        Get the compile scorecard job from the YAML associated with these parameters.

        parameters:
            path: Applicable scorecard path
            model_id: The ID of the QAIHM model being tested
            device: The targeted device
            component: The name of the model component being tested, if applicable
        """
        return CompileScorecardJob(
            component or model_id,
            self.get_job_id(
                path, model_id, device, component, fallback_to_universal_device=True
            ),
            device,
            None,
            path,
        )


class ProfileScorecardJobYaml(ScorecardJobYaml):
    def get_job(
        self,
        path: ScorecardProfilePath,
        model_id: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ) -> ProfileScorecardJob:
        """
        Get the profile scorecard job from the YAML associated with these parameters.

        parameters:
            path: Applicable scorecard path
            model_id: The ID of the QAIHM model being tested
            device: The targeted device
            component: The name of the model component being tested, if applicable
        """
        return ProfileScorecardJob(
            component or model_id,
            self.get_job_id(path, model_id, device, component),
            device,
            None,
            path,
        )
