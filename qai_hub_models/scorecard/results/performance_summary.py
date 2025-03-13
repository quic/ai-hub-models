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
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.results.chipset_helpers import (
    chipset_marketing_name,
    get_supported_devices,
    supported_chipsets_santized,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    ProfileScorecardJob,
    QuantizeScorecardJob,
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
    scorecard_job_type: type[ScorecardJobTypeVar]

    def __init__(
        self,
        model_id: str,
        device: ScorecardDevice,
        run_per_path: dict[
            ScorecardPathTypeVar, ScorecardJobTypeVar
        ],  # Map<path, Summary>
    ):
        self.model_id = model_id
        self.device = device
        self.run_per_path: dict[
            ScorecardPathTypeVar, ScorecardJobTypeVar
        ] = run_per_path

    @classmethod
    def from_runs(
        cls: type[_DeviceSummaryTypeVar],
        model_id: str,
        device: ScorecardDevice,
        path_runs: list[ScorecardJobTypeVar],
    ):
        # Figure out unique devices in various baselines
        run_per_path: dict[ScorecardPathTypeVar, ScorecardJobTypeVar] = {}
        for run in path_runs:
            assert run._device == device  # Device should match
            run_per_path[run.path] = run  # type: ignore[index]

        return cls(model_id, device, run_per_path)

    def get_run(self, path: ScorecardPathTypeVar) -> ScorecardJobTypeVar:
        if x := self.run_per_path.get(path):
            return x

        # Create a "Skipped" run to return
        return self.__class__.scorecard_job_type(
            self.model_id, None, self.device, False, None, path  # type: ignore[arg-type]
        )


_DeviceSummaryTypeVar = TypeVar("_DeviceSummaryTypeVar", bound=ScorecardDeviceSummary)
# Specific typevar. Autofill has trouble resolving types for nested generics without specifically listing ineritors of the generic base.
DeviceSummaryTypeVar = TypeVar(
    "DeviceSummaryTypeVar",
    "DevicePerfSummary",
    "DeviceQuantizeSummary",
    "DeviceCompileSummary",
    "DeviceInferenceSummary",
)


class ScorecardModelSummary(
    Generic[DeviceSummaryTypeVar, ScorecardJobTypeVar, ScorecardPathTypeVar]
):
    device_summary_type: type[DeviceSummaryTypeVar]
    scorecard_job_type: type[ScorecardJobTypeVar]

    @property
    def component_ids(self) -> Iterable[str]:
        """
        Components mapped by this summary.

        If there are no model components, returns [model_id].
        model_id is a valid component name for looking up runs in summaries for models without components.
        """
        return self.runs_per_component_device.keys()

    def is_same_model(self, other: ScorecardModelSummary) -> bool:
        """Returns true if this summary and the provided summary map to the same model definition."""
        return self.model_id == other.model_id and set(self.component_ids) == set(
            other.component_ids
        )

    def __init__(
        self,
        model_id: str,
        runs_per_device: dict[ScorecardDevice, DeviceSummaryTypeVar] | None = None,
        runs_per_component_device: dict[
            str, dict[ScorecardDevice, DeviceSummaryTypeVar]
        ]
        | None = None,
    ):
        """
        Create a Summary for a single Scorecard Model.

        Parameters:
            model_id: str
                Model ID.

            runs_per_device: dict[ScorecardDevice, DeviceSummaryTypeVar] | None
                Set if the model does not have components.

            runs_per_component_device: dict[component_id, dict[ScorecardDevice, DeviceSummaryTypeVar]] | None
                Set if the model has components.
        """
        if (runs_per_device is None) == (runs_per_component_device is None):
            raise ValueError(
                "Either runs_per_device or runs_per_component_device must be set, but not both."
            )

        self.model_id = model_id
        self.has_components = runs_per_component_device is not None
        self.runs_per_component_device: dict[
            str, dict[ScorecardDevice, DeviceSummaryTypeVar]
        ]
        if not self.has_components:
            # To make writing helper functions easier, models without components are stored as map with one entry:
            #   model_id: runs_per_device
            assert runs_per_device is not None
            self.runs_per_component_device = {model_id: runs_per_device}
        else:
            assert runs_per_component_device is not None
            self.runs_per_component_device = runs_per_component_device

    @classmethod
    def from_runs(
        cls: type[_ModelSummaryTypeVar],
        model_id: str,
        path_runs: list[ScorecardJobTypeVar],
        components: list[str] | None = None,
    ):
        summaries_per_device_component: dict[
            str, dict[ScorecardDevice, DeviceSummaryTypeVar]
        ] = {}
        component_ids = [model_id] if components is None else components
        for component_id in component_ids:
            component_dict: dict[ScorecardDevice, list[ScorecardJobTypeVar]] = {}
            for run in path_runs:
                if run.model_id == component_id:
                    job_list = component_dict.get(run._device, list())
                    component_dict[run._device] = job_list
                    job_list.append(run)
            summaries_per_device_component[component_id] = {
                device: cls.device_summary_type.from_runs(model_id, device, runs)
                for device, runs in component_dict.items()
            }

        if components is None:
            return cls(model_id, summaries_per_device_component[model_id])
        else:
            return cls(model_id, None, summaries_per_device_component)

    def get_run(
        self,
        device: ScorecardDevice,
        path: ScorecardPathTypeVar,
        component: str | None = None,
    ) -> ScorecardJobTypeVar:
        """
        Get a scorecard job matching these parameters.

        Parameters:
            device: ScorecardDevice
            path: ScorecardPathTypeVar
            component: str | None
                To make writing helper functions easier, users may pass component == model_id if this model does not have components.
        """
        if component:
            if not self.has_components and component != self.model_id:
                raise ValueError(
                    "Cannot provide component name for models without components."
                )
        elif self.has_components:
            raise ValueError("Must provide component name for models with components.")

        if component_device_map := self.runs_per_component_device.get(
            component or self.model_id
        ):
            if summary := component_device_map.get(device):
                return summary.get_run(path)  # type: ignore[arg-type,return-value]

        # Create a "Skipped" run to return
        return self.__class__.scorecard_job_type(
            self.model_id, None, device, False, None, path  # type: ignore[arg-type]
        )


_ModelSummaryTypeVar = TypeVar("_ModelSummaryTypeVar", bound=ScorecardModelSummary)
# Specific typevar. Autofill has trouble resolving types for nested generics without specifically listing ineritors of the generic base.
ModelSummaryTypeVar = TypeVar(
    "ModelSummaryTypeVar",
    "ModelPerfSummary",
    "ModelQuantizeSummary",
    "ModelCompileSummary",
    "ModelInferenceSummary",
)


class DevicePerfSummary(
    ScorecardDeviceSummary[ProfileScorecardJob, ScorecardProfilePath]
):
    scorecard_job_type = ProfileScorecardJob

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


class ModelPerfSummary(
    ScorecardModelSummary[DevicePerfSummary, ProfileScorecardJob, ScorecardProfilePath]
):
    device_summary_type = DevicePerfSummary
    scorecard_job_type = ProfileScorecardJob

    def get_universal_assets(
        self,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
        component: str | None = None,
    ):
        assert (
            component is not None
        ) == self.has_components or component == self.model_id

        universal_assets = {}
        for path in ScorecardProfilePath:
            if not path.compile_path.is_universal or path in exclude_paths:
                continue

            # Only add a universal asset if at least 1 profile job succeeded.
            for runs_per_device in self.runs_per_component_device[
                component or self.model_id
            ].values():
                path_run = runs_per_device.run_per_path.get(path, None)
                if path_run and path_run.success:
                    universal_assets[path.long_name] = path_run.job.model.model_id

        return universal_assets

    def get_chipsets(
        self,
        include_internal_devices: bool = False,
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
    ) -> set[str]:
        chips_by_component: dict[str, set[str]] = dict()
        for component_id, summary_by_device in self.runs_per_component_device.items():
            chips: set[str] = set()
            chips_by_component[component_id] = chips
            for device, device_summary in summary_by_device.items():
                # At least 1 successful run must exist for this chipset
                success = False
                for run in device_summary.run_per_path.values():
                    if run.success:
                        success = True
                        break
                if not success:
                    continue

                # Don't include private devices
                if (
                    not include_internal_devices
                    and not device.public
                    and device.form_factor not in exclude_form_factors
                ):
                    continue

                chips.add(device.chipset)

        # Supported chipsets for this model must be supported by all model components
        out: set[str] = next(iter(chips_by_component.values()))
        if len(chips_by_component) > 1:
            for chips in chips_by_component.values():
                out = out.intersection(chips)

        return out

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        include_internal_devices: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
        model_name: str | None = None,
    ) -> dict[str, str | list[Any] | dict[str, Any]]:
        perf_card: dict[str, str | list[Any] | dict[str, Any]] = {}

        chips = self.get_chipsets(include_internal_devices, exclude_form_factors)
        perf_card["aggregated"] = dict(
            supported_devices=get_supported_devices(chips),
            supported_chipsets=supported_chipsets_santized(chips),
        )

        components_list: list[dict[str, Any]] = []
        for component_id, summary_per_device in self.runs_per_component_device.items():
            component_perf_card: list[dict[str, Union[str, dict[str, str]]]] = []
            for summary in summary_per_device.values():
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
                        component_perf_card.append(device_summary)

            components_list.append(
                {
                    "name": component_id
                    if component_id != self.model_id
                    else model_name,
                    "universal_assets": self.get_universal_assets(
                        exclude_paths=exclude_paths,
                        component=component_id if self.has_components else None,
                    ),
                    "performance_metrics": component_perf_card,
                }
            )

        perf_card["models"] = components_list
        return perf_card

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


class DeviceQuantizeSummary(
    ScorecardDeviceSummary[QuantizeScorecardJob, ScorecardCompilePath]
):
    scorecard_job_type = QuantizeScorecardJob

    def get_run(self, path: ScorecardCompilePath) -> QuantizeScorecardJob:
        return super().get_run(ScorecardCompilePath.ONNX)


class DeviceCompileSummary(
    ScorecardDeviceSummary[CompileScorecardJob, ScorecardCompilePath]
):
    scorecard_job_type = CompileScorecardJob


class ModelQuantizeSummary(
    ScorecardModelSummary[
        DeviceQuantizeSummary, QuantizeScorecardJob, ScorecardCompilePath
    ]
):
    device_summary_type = DeviceQuantizeSummary
    scorecard_job_type = QuantizeScorecardJob

    def get_run(
        self,
        device: ScorecardDevice,
        path: ScorecardCompilePath,
        component: str | None = None,
    ) -> QuantizeScorecardJob:
        return super().get_run(cs_universal, ScorecardCompilePath.ONNX, component)


class ModelCompileSummary(
    ScorecardModelSummary[
        DeviceCompileSummary, CompileScorecardJob, ScorecardCompilePath
    ]
):
    device_summary_type = DeviceCompileSummary
    scorecard_job_type = CompileScorecardJob

    def get_run(
        self,
        device: ScorecardDevice,
        path: ScorecardCompilePath,
        component: str | None = None,
        universal_device_fallback: bool = True,
    ) -> CompileScorecardJob:
        run = super().get_run(device, path, component)
        if (
            universal_device_fallback
            and run.skipped
            and path not in device.compile_paths
        ):
            run = super().get_run(cs_universal, path, component)
        return run


class DeviceInferenceSummary(
    ScorecardDeviceSummary[InferenceScorecardJob, ScorecardProfilePath]
):
    scorecard_job_type = InferenceScorecardJob


class ModelInferenceSummary(
    ScorecardModelSummary[
        DeviceInferenceSummary, InferenceScorecardJob, ScorecardProfilePath
    ]
):
    device_summary_type = DeviceInferenceSummary
    scorecard_job_type = InferenceScorecardJob
