# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pprint
from collections.abc import Iterable
from typing import Any, Generic, TypeVar, Union

from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.results.chipset_helpers import (
    chipset_marketing_name,
    get_supported_devices,
    supported_chipsets_santized,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    ProfileScorecardJob,
    ScorecardJobTypeVar,
    ScorecardPathTypeVar,
)

# Caching this information is helpful because it requires pulling data from hub.
# Pulling data from hub is slow.
__REFERENCE_DEVICE_INFO_PER_CHIPSET = {}


def get_reference_device_info(device: ScorecardDevice) -> dict[str, str]:
    chipset = device.chipset
    if chipset not in __REFERENCE_DEVICE_INFO_PER_CHIPSET:
        hub_device = device.reference_device
        device_name = hub_device.name
        os_version = hub_device.os
        os_name, form_factor, manufacturer = "", "", ""
        for attr in hub_device.attributes:
            if attr.startswith("vendor"):
                manufacturer = attr.split(":")[-1]
            if attr.startswith("format"):
                form_factor = attr.split(":")[-1]
            if attr.startswith("os"):
                os_name = attr.split(":")[-1].capitalize()
        chipset = chipset_marketing_name(chipset)
        __REFERENCE_DEVICE_INFO_PER_CHIPSET[chipset] = dict(
            name=device_name,
            os=os_version,
            form_factor=form_factor.capitalize(),
            os_name=os_name,
            manufacturer=manufacturer.capitalize(),
            chipset=chipset,
        )
    return __REFERENCE_DEVICE_INFO_PER_CHIPSET[chipset]


class ScorecardDeviceSummary(Generic[ScorecardJobTypeVar, ScorecardPathTypeVar]):
    def __init__(
        self,
        device: ScorecardDevice,
        run_per_path: dict[
            ScorecardPathTypeVar, ScorecardJobTypeVar
        ],  # Map<path, Summary>
    ):
        self.device = device
        self.run_per_path: dict[
            ScorecardPathTypeVar, ScorecardJobTypeVar
        ] = run_per_path

    @classmethod
    def from_runs(
        cls: type[_DeviceSummaryTypeVar],
        device: ScorecardDevice,
        path_runs: list[ScorecardJobTypeVar],
    ):
        # Figure out unique devices in various baselines
        run_per_path: dict[ScorecardPathTypeVar, ScorecardJobTypeVar] = {}
        for run in path_runs:
            assert run._device == device  # Device should match
            run_per_path[run.path] = run  # type: ignore

        return cls(device, run_per_path)


_DeviceSummaryTypeVar = TypeVar("_DeviceSummaryTypeVar", bound=ScorecardDeviceSummary)
# Specific typevar. Autofill has trouble resolving types for nested generics without specifically listing ineritors of the generic base.
DeviceSummaryTypeVar = TypeVar(
    "DeviceSummaryTypeVar",
    "DevicePerfSummary",
    "DeviceCompileSummary",
    "DeviceInferenceSummary",
)


class ScorecardModelSummary(Generic[DeviceSummaryTypeVar, ScorecardJobTypeVar]):
    device_summary_type: type[DeviceSummaryTypeVar]

    def __init__(
        self,
        model_id: str,
        runs_per_device: dict[ScorecardDevice, DeviceSummaryTypeVar],
    ):
        self.model_id = model_id
        self.runs_per_device: dict[
            ScorecardDevice, DeviceSummaryTypeVar
        ] = runs_per_device

    @classmethod
    def from_runs(
        cls: type[_ModelSummaryTypeVar],
        model_id: str,
        path_runs: list[ScorecardJobTypeVar],
    ):
        runs_per_device: dict[ScorecardDevice, list[ScorecardJobTypeVar]] = {}
        for run in path_runs:
            assert run.model_id == model_id  # model id should match
            list = runs_per_device.get(run._device, [])
            runs_per_device[run._device] = list
            list.append(run)

        return cls(
            model_id,
            {
                device: cls.device_summary_type.from_runs(device, runs)
                for device, runs in runs_per_device.items()
            },
        )


_ModelSummaryTypeVar = TypeVar("_ModelSummaryTypeVar", bound=ScorecardModelSummary)
# Specific typevar. Autofill has trouble resolving types for nested generics without specifically listing ineritors of the generic base.
ModelSummaryTypeVar = TypeVar(
    "ModelSummaryTypeVar",
    "ModelPerfSummary",
    "ModelCompileSummary",
    "ModelInferenceSummary",
)


class ScorecardSummary(Generic[ModelSummaryTypeVar, ScorecardJobTypeVar]):
    model_summary_type: type[ModelSummaryTypeVar]

    def __init__(self, runs_per_model: dict[str, ModelSummaryTypeVar]):
        self.runs_per_model: dict[str, ModelSummaryTypeVar] = runs_per_model

    @classmethod
    def from_runs(
        cls: type[_ScorecardSummaryTypeVar], model_runs: list[ScorecardJobTypeVar]
    ) -> _ScorecardSummaryTypeVar:
        # Figure out unique models in various baselines
        runs_per_model: dict[str, list[ScorecardJobTypeVar]] = {}
        for run in model_runs:
            list = runs_per_model.get(run.model_id, [])
            list.append(run)
            runs_per_model[run.model_id] = list

        return cls(
            {
                model_id: cls.model_summary_type.from_runs(model_id, runs)
                for model_id, runs in runs_per_model.items()
            }
        )


_ScorecardSummaryTypeVar = TypeVar("_ScorecardSummaryTypeVar", bound=ScorecardSummary)
# Specific typevar. Autofill has trouble resolving types for nested generics without specifically listing ineritors of the generic base.
ScorecardSummaryTypeVar = TypeVar(
    "ScorecardSummaryTypeVar", "CompileSummary", "PerfSummary", "InferenceSummary"
)


class DevicePerfSummary(
    ScorecardDeviceSummary[ProfileScorecardJob, ScorecardProfilePath]
):
    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
    ) -> dict[str, str | dict[str, str]]:
        perf_card: dict[str, str | dict[str, str]] = {}
        max_date = None
        for path, run in self.run_per_path.items():
            if (
                not run.skipped  # Skipped runs are not included
                and path
                not in exclude_paths  # exclude paths that the user does not want included
                and (
                    include_failed_jobs or not run.failed
                )  # exclude failed jobs if requested
            ):
                perf_card[path.long_name] = run.performance_metrics
                if max_date is None:
                    max_date = run.date
                elif run.date is not None:
                    max_date = max(max_date, run.date)
        if not perf_card:
            return {}
        perf_card["reference_device_info"] = get_reference_device_info(self.device)
        # The timestamp for the device is the latest creation time among the runs
        # If max_date is still None for some reason, something went wrong
        assert max_date is not None
        perf_card["timestamp"] = max_date.isoformat() + "Z"
        return perf_card

    def __repr__(self) -> str:
        return pprint.pformat(self.get_perf_card())


class ModelPerfSummary(ScorecardModelSummary[DevicePerfSummary, ProfileScorecardJob]):
    device_summary_type = DevicePerfSummary

    def get_universal_assets(self, exclude_paths: Iterable[ScorecardProfilePath] = []):
        universal_assets = {}
        for path in ScorecardProfilePath:
            if not path.compile_path.is_universal or path in exclude_paths:
                continue

            # Only add a universal asset if at least 1 profile job succeeded.
            for runs_per_device in self.runs_per_device.values():
                path_run = runs_per_device.run_per_path.get(path, None)
                if path_run and path_run.success:
                    universal_assets[path.long_name] = path_run.job.model.model_id

        return universal_assets

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        include_internal_devices: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
    ) -> list[dict[str, Union[str, dict[str, str]]]]:
        perf_card = []
        for summary in self.runs_per_device.values():
            if (
                include_internal_devices
                or summary.device.public
                and summary.device.form_factor not in exclude_form_factors
            ):
                device_summary = summary.get_perf_card(
                    include_failed_jobs, exclude_paths
                )

                # If device had no runs, omit it from the card
                if len(device_summary) != 0:
                    perf_card.append(device_summary)
        return perf_card

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


class PerfSummary(ScorecardSummary[ModelPerfSummary, ProfileScorecardJob]):
    model_summary_type = ModelPerfSummary

    def get_chipsets(
        self,
        include_internal_devices: bool = False,
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
    ) -> set[str]:
        chips: set[str] = set()
        for model_id, model_summary in self.runs_per_model.items():
            for device, device_summary in model_summary.runs_per_device.items():
                # At least 1 successful run must exist for this chipset
                success = False
                for run in device_summary.run_per_path.values():
                    if run.success:
                        success = True
                        break
                if not success:
                    continue

                # Don't incude disabled models
                if model_id in device.disabled_models:
                    continue

                # Don't include private devices
                if (
                    not include_internal_devices
                    and not device.public
                    and device.form_factor not in exclude_form_factors
                ):
                    continue

                chips.add(device.chipset)
        return chips

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        include_internal_devices: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
    ) -> dict[str, str | list[Any] | dict[str, Any]]:
        perf_card: dict[str, str | list[Any] | dict[str, Any]] = {}

        chips = self.get_chipsets(include_internal_devices, exclude_form_factors)
        perf_card["aggregated"] = dict(
            supported_devices=get_supported_devices(chips),
            supported_chipsets=supported_chipsets_santized(chips),
        )

        models_list: list[dict[str, Any]] = []
        for model_id, summary in self.runs_per_model.items():
            models_list.append(
                {
                    "name": model_id,
                    "universal_assets": summary.get_universal_assets(
                        exclude_paths=exclude_paths
                    ),
                    "performance_metrics": summary.get_perf_card(
                        include_failed_jobs,
                        include_internal_devices,
                        exclude_paths,
                        exclude_form_factors,
                    ),
                }
            )
        perf_card["models"] = models_list
        return perf_card

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


class DeviceCompileSummary(
    ScorecardDeviceSummary[CompileScorecardJob, ScorecardCompilePath]
):
    pass


class ModelCompileSummary(
    ScorecardModelSummary[DeviceCompileSummary, CompileScorecardJob]
):
    device_summary_type = DeviceCompileSummary


class CompileSummary(ScorecardSummary[ModelCompileSummary, CompileScorecardJob]):
    model_summary_type = ModelCompileSummary


class DeviceInferenceSummary(
    ScorecardDeviceSummary[InferenceScorecardJob, ScorecardProfilePath]
):
    pass


class ModelInferenceSummary(
    ScorecardModelSummary[DeviceInferenceSummary, InferenceScorecardJob]
):
    device_summary_type = DeviceInferenceSummary


class InferenceSummary(ScorecardSummary[ModelInferenceSummary, InferenceScorecardJob]):
    model_summary_type = ModelInferenceSummary
