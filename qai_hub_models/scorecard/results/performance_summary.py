# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pprint
from collections.abc import Iterable
from typing import Generic, TypeVar

from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.results.chipset_helpers import (
    get_supported_devices,
    sorted_chipsets,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    ProfileScorecardJob,
    QuantizeScorecardJob,
    ScorecardJobTypeVar,
    ScorecardPathOrNoneTypeVar,
)

# This file defines summary mappings for scorecard jobs.
# This is the hierarchy of base classes:
#
# Python Dict
# Summary mapping for a single model ID.
#   map<model_id: ScorecardModelSummary>
#
#       ScorecardModelSummary
#         Summary for all precisions available for a model.
#         map<Precision: ScorecardModelPrecisionSummary>
#
#            ScorecardModelPrecisionSummary
#              Summary for all model components with a specific QDQ spec.
#              (If no components, a single "component name" is stored in the map.
#               This "component name" is the model ID.)
#              map<Component Name: map<ScorecardDevice: ScorecardDeviceSummary>>
#
#                   ScorecardDeviceSummary
#                     Summary for one model component targeting a specific QDQ spec and device.
#                     map<ScorecardPath: ScorecardJob>
#
#                         ScorecardJob
#                           Job for one scorecard path / device / model component / QDQ spec / model ID.
#
#
# Each base class has one child class per supported job type. The supported job types are:
#   QuantizeJob (only ScorecardCompilePath.ONNX is supported)
#   CompileJob (mapped by ScorecardCompilePath)
#   ProfileJob (mapped by ScorecardProfilePath)
#   InferenceJob (mapped by ScorecardProfilePath)


class ScorecardDeviceSummary(Generic[ScorecardJobTypeVar, ScorecardPathOrNoneTypeVar]):
    scorecard_job_type: type[ScorecardJobTypeVar]

    def __init__(
        self,
        model_id: str,
        precision: Precision,
        device: ScorecardDevice,
        run_per_path: dict[
            ScorecardPathOrNoneTypeVar, ScorecardJobTypeVar
        ],  # Map<path, Summary>
    ):
        self.model_id = model_id
        self.precision = precision
        self.device = device
        self.run_per_path: dict[
            ScorecardPathOrNoneTypeVar, ScorecardJobTypeVar
        ] = run_per_path

    @classmethod
    def from_runs(
        cls: type[_DeviceSummaryTypeVar],
        model_id: str,
        precision: Precision,
        device: ScorecardDevice,
        path_runs: list[ScorecardJobTypeVar],
    ):
        # Figure out unique devices in various baselines
        run_per_path: dict[ScorecardPathOrNoneTypeVar, ScorecardJobTypeVar] = {}
        for run in path_runs:
            assert run._device == device  # Device should match
            run_per_path[run.path] = run  # type: ignore[index]

        return cls(model_id, precision, device, run_per_path)

    def get_run(self, path: ScorecardPathOrNoneTypeVar) -> ScorecardJobTypeVar:
        if x := self.run_per_path.get(path):
            return x

        # Create a "Skipped" run to return
        return self.__class__.scorecard_job_type(
            self.model_id, self.precision, None, self.device, False, None, path  # type: ignore[arg-type]
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


class ScorecardModelPrecisionSummary(
    Generic[DeviceSummaryTypeVar, ScorecardJobTypeVar, ScorecardPathOrNoneTypeVar]
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

    def __init__(
        self,
        model_id: str,
        precision: Precision,
        runs_per_device: dict[ScorecardDevice, DeviceSummaryTypeVar] | None = None,
        runs_per_component_device: dict[
            str, dict[ScorecardDevice, DeviceSummaryTypeVar]
        ]
        | None = None,
    ):
        """
        Create a Summary for a Scorecard Model with a specific Precision.

        Parameters:
            model_id: str
                Model ID.

            precision: Precision
                Model quantization scheme.

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
        self.precision = precision
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
        cls: type[_ModelPrecisionSummaryTypeVar],
        model_id: str,
        precision: Precision,
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
                device: cls.device_summary_type.from_runs(
                    model_id, precision, device, runs
                )
                for device, runs in component_dict.items()
            }

        if components is None:
            return cls(model_id, precision, summaries_per_device_component[model_id])
        else:
            return cls(model_id, precision, None, summaries_per_device_component)

    def get_run(
        self,
        device: ScorecardDevice,
        path: ScorecardPathOrNoneTypeVar,
        component: str | None = None,
    ) -> ScorecardJobTypeVar:
        """
        Get a scorecard job matching these parameters.

        Parameters:
            device: ScorecardDevice
            path: ScorecardPathOrNoneTypeVar
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
            self.model_id, self.precision, None, device, False, None, path  # type: ignore[arg-type]
        )


_ModelPrecisionSummaryTypeVar = TypeVar(
    "_ModelPrecisionSummaryTypeVar", bound=ScorecardModelPrecisionSummary
)
# Specific typevar. Autofill has trouble resolving types for nested generics without specifically listing ineritors of the generic base.
ModelPrecisionSummaryTypeVar = TypeVar(
    "ModelPrecisionSummaryTypeVar",
    "ModelPrecisionPerfSummary",
    "ModelPrecisionQuantizeSummary",
    "ModelPrecisionCompileSummary",
    "ModelPrecisionInferenceSummary",
)


class ScorecardModelSummary(
    Generic[
        ModelPrecisionSummaryTypeVar, ScorecardJobTypeVar, ScorecardPathOrNoneTypeVar
    ]
):
    model_summary_type: type[ModelPrecisionSummaryTypeVar]
    scorecard_job_type: type[ScorecardJobTypeVar]

    def __init__(
        self,
        model_id: str = "UNKNOWN",
        summaries_per_precision: dict[Precision, ModelPrecisionSummaryTypeVar] = {},
    ):
        """
        Create a Summary for a single Scorecard Model.

        Parameters:
            model_id: str
                Model ID.

            summaries_per_precision: dict[Precision, ModelPrecisionSummaryTypeVar]
                Summary per precision.
        """
        self.model_id = model_id
        self.summaries_per_precision: dict[
            Precision, ModelPrecisionSummaryTypeVar
        ] = summaries_per_precision

    @classmethod
    def from_runs(
        cls: type[_ModelSummaryTypeVar],
        model_id: str,
        runs: list[ScorecardJobTypeVar],
        components: list[str] | None = None,
    ):
        summaries_per_precision: dict[Precision, list[ScorecardJobTypeVar]] = {}
        for run in runs:
            if run.precision in summaries_per_precision:
                summaries_per_precision[run.precision].append(run)
            else:
                summaries_per_precision[run.precision] = [run]

        return cls(
            model_id,
            {
                precision: cls.model_summary_type.from_runs(
                    model_id, precision, runs, components
                )
                for precision, runs in summaries_per_precision.items()
            },
        )

    def get_run(
        self,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardPathOrNoneTypeVar,
        component: str | None = None,
    ) -> ScorecardJobTypeVar:
        """
        Get a scorecard job matching these parameters.

        Parameters:
            device: ScorecardDevice
            path: ScorecardPathOrNoneTypeVar
            component: str | None
                To make writing helper functions easier, users may pass component == model_id if this model does not have components.
        """
        if model_summary := self.summaries_per_precision.get(precision):
            return model_summary.get_run(device, path, component)  # type: ignore[arg-type,return-value]

        # Create a "Skipped" run to return
        return self.__class__.scorecard_job_type(
            self.model_id, precision, None, device, False, None, path  # type: ignore[arg-type]
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


# --------------------------------------
#
# Profile Job Summary Classes
#


class DevicePerfSummary(
    ScorecardDeviceSummary[ProfileScorecardJob, ScorecardProfilePath]
):
    scorecard_job_type = ProfileScorecardJob

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
    ) -> dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails]:
        perf_card: dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails] = {}
        for path, run in self.run_per_path.items():
            if (
                not run.skipped  # Skipped runs are not included
                and path
                not in exclude_paths  # exclude paths that the user does not want included
                and (
                    include_failed_jobs or not run.failed
                )  # exclude failed jobs if requested
            ):
                perf_card[path] = run.performance_metrics
        return perf_card

    def __repr__(self) -> str:
        return pprint.pformat(self.get_perf_card())


class ModelPrecisionPerfSummary(
    ScorecardModelPrecisionSummary[
        DevicePerfSummary, ProfileScorecardJob, ScorecardProfilePath
    ]
):
    device_summary_type = DevicePerfSummary
    scorecard_job_type = ProfileScorecardJob

    def get_target_assets(
        self,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
        component: str | None = None,
    ) -> tuple[
        dict[ScorecardProfilePath, str],
        dict[ScorecardDevice, dict[ScorecardProfilePath, str]],
    ]:
        universal_assets: dict[ScorecardProfilePath, str] = {}
        device_assets: dict[ScorecardDevice, dict[ScorecardProfilePath, str]] = {}
        for runs_per_device in self.runs_per_component_device[
            component or self.model_id
        ].values():
            if runs_per_device.device.form_factor in exclude_form_factors:
                continue
            for path, path_run in runs_per_device.run_per_path.items():
                if path in exclude_paths or not path_run.success:
                    continue
                if path.compile_path.is_universal:
                    if path not in universal_assets:
                        universal_assets[path] = path_run.job.model.model_id
                else:
                    if runs_per_device.device not in device_assets:
                        device_assets[runs_per_device.device] = {}
                    device_assets[runs_per_device.device][
                        path
                    ] = path_run.job.model.model_id

        return universal_assets, device_assets

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        include_internal_devices: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
        model_name: str | None = None,
    ) -> QAIHMModelPerf.PrecisionDetails:
        components: dict[str, QAIHMModelPerf.ComponentDetails] = {}
        for component_id, summary_per_device in self.runs_per_component_device.items():
            component_perf_card: dict[
                ScorecardDevice,
                dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
            ] = {}
            for device, summary in sorted(
                summary_per_device.items(), key=lambda dk: dk[0].reference_device_name
            ):
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
                        component_perf_card[device] = device_summary

            component_name = (
                component_id if component_id != self.model_id else (model_name or "")
            )
            universal_assets, device_assets = self.get_target_assets(
                exclude_paths, exclude_form_factors, component_id
            )
            components[component_name] = QAIHMModelPerf.ComponentDetails(
                universal_assets=universal_assets,
                device_assets=device_assets,
                performance_metrics=component_perf_card,
            )

        # Remove paths that aren't supported by all components.
        # A path must support all components for it to "show up" in perf yaml
        num_components = len(components)
        if not include_failed_jobs and num_components > 1:
            # Map<Device, Map<Path, Count of Components that support this device + path>>
            supported_device_runtimes: dict[
                ScorecardDevice, dict[ScorecardProfilePath, int]
            ] = {}

            # Walk each component and add to the counter for each device + runtime support pairing.
            for component in components.values():
                for device, runtime_dict in component.performance_metrics.items():
                    if device not in supported_device_runtimes:
                        supported_device_runtimes[device] = {}
                    for runtime in runtime_dict:
                        if runtime not in supported_device_runtimes[device]:
                            supported_device_runtimes[device][runtime] = 0
                        supported_device_runtimes[device][runtime] += 1

            # Walk each component and remove component + device pairs that aren't supported by all components.
            for component in components.values():
                devices = list(component.performance_metrics.keys())
                for device in devices:
                    runtimes = list(component.performance_metrics[device].keys())

                    # Remove runtime + device pairs that aren't supported by all components.
                    for runtime in runtimes:
                        if supported_device_runtimes[device][runtime] != num_components:
                            component.performance_metrics[device].pop(runtime)

                    # Remove the device entirely if all runtimes were removed.
                    if not component.performance_metrics[device]:
                        component.performance_metrics.pop(device)
                        if device in component.device_assets:
                            component.device_assets.pop(device)

            # Remove universal assets for runtimes that were entirely removed
            # because they aren't supported by all components.
            for component in components.values():
                universal_runtimes = {r: False for r in component.universal_assets}
                for runtime_dict in component.performance_metrics.values():
                    for runtime in runtime_dict:
                        if runtime in universal_runtimes:
                            universal_runtimes[runtime] = True
                    if all(universal_runtimes.values()):
                        break

                for runtime, runtime_exists in universal_runtimes.items():
                    if not runtime_exists:
                        component.universal_assets.pop(runtime)

        # Remove components with no jobs.
        component_names = list(components.keys())
        for component_name in component_names:
            if not components[component_name].performance_metrics:
                components.pop(component_name)

        return QAIHMModelPerf.PrecisionDetails(components=components)

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


class ModelPerfSummary(
    ScorecardModelSummary[
        ModelPrecisionPerfSummary, ProfileScorecardJob, ScorecardProfilePath
    ]
):
    model_summary_type = ModelPrecisionPerfSummary
    scorecard_job_type = ProfileScorecardJob

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        include_internal_devices: bool = True,
        exclude_paths: dict[Precision, list[ScorecardProfilePath]] = {},
        exclude_form_factors: Iterable[ScorecardDevice.FormFactor] = [],
        model_name: str | None = None,
    ) -> QAIHMModelPerf:

        precision_cards = {
            p: s.get_perf_card(
                include_failed_jobs,
                include_internal_devices,
                exclude_paths.get(p, []),
                exclude_form_factors,
                model_name,
            )
            for p, s in sorted(
                self.summaries_per_precision.items(),
                # Sort by precision name
                key=lambda ps: str(ps[0]),
            )
        }

        # Remove precisions with no jobs.
        for p in self.summaries_per_precision.keys():
            if not precision_cards[p].components or all(
                not component_card.performance_metrics
                for component_card in precision_cards[p].components.values()
            ):
                precision_cards.pop(p)

        supported_chipsets: set[str] = set()
        for precision_card in precision_cards.values():
            if include_failed_jobs:
                # If failed jobs are included, we have to loop though everything and find what jobs ran successfully.
                for component_card in precision_card.components.values():
                    for (
                        device,
                        runs_by_runtime,
                    ) in component_card.performance_metrics.items():
                        for run in runs_by_runtime.values():
                            if run.inference_time_milliseconds is not None:
                                supported_chipsets.update(
                                    device.extended_supported_chipsets
                                )
                                break
            else:
                # If failed jobs aren't included, all components will have the same set of
                # supported devices / runtimes, and all jobs will have succeeded.
                for device in next(
                    iter(precision_card.components.values())
                ).performance_metrics:
                    supported_chipsets.update(device.extended_supported_chipsets)

        return QAIHMModelPerf(
            supported_devices=get_supported_devices(supported_chipsets),
            supported_chipsets=sorted_chipsets(supported_chipsets),
            precisions=precision_cards,
        )

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


# --------------------------------------
#
# Quantize Job Summary Classes
#


class DeviceQuantizeSummary(ScorecardDeviceSummary[QuantizeScorecardJob, None]):
    scorecard_job_type = QuantizeScorecardJob

    def get_run(
        self, path: ScorecardCompilePath | ScorecardProfilePath | None
    ) -> QuantizeScorecardJob:
        return super().get_run(None)


class ModelPrecisionQuantizeSummary(
    ScorecardModelPrecisionSummary[DeviceQuantizeSummary, QuantizeScorecardJob, None]
):
    device_summary_type = DeviceQuantizeSummary
    scorecard_job_type = QuantizeScorecardJob

    def get_run(
        self,
        device: ScorecardDevice,
        path: ScorecardCompilePath | ScorecardProfilePath | None,
        component: str | None = None,
    ) -> QuantizeScorecardJob:
        return super().get_run(cs_universal, None, component)


class ModelQuantizeSummary(
    ScorecardModelSummary[ModelPrecisionQuantizeSummary, InferenceScorecardJob, None]
):
    model_summary_type = ModelPrecisionQuantizeSummary
    scorecard_job_type = InferenceScorecardJob


# --------------------------------------
#
# Compile Job Summary Classes
#


class DeviceCompileSummary(
    ScorecardDeviceSummary[CompileScorecardJob, ScorecardCompilePath]
):
    scorecard_job_type = CompileScorecardJob

    def get_run(
        self, path: ScorecardCompilePath | ScorecardProfilePath
    ) -> CompileScorecardJob:
        if isinstance(path, ScorecardProfilePath):
            path = path.compile_path
        return super().get_run(path)


class ModelPrecisionCompileSummary(
    ScorecardModelPrecisionSummary[
        DeviceCompileSummary, CompileScorecardJob, ScorecardCompilePath
    ]
):
    device_summary_type = DeviceCompileSummary
    scorecard_job_type = CompileScorecardJob

    def get_run(
        self,
        device: ScorecardDevice,
        path: ScorecardCompilePath | ScorecardProfilePath,
        component: str | None = None,
        universal_device_fallback: bool = True,
    ) -> CompileScorecardJob:
        if isinstance(path, ScorecardProfilePath):
            path = path.compile_path
        run = super().get_run(device, path, component)
        if (
            universal_device_fallback
            and run.skipped
            and path not in device.compile_paths
        ):
            run = super().get_run(cs_universal, path, component)
        return run


class ModelCompileSummary(
    ScorecardModelSummary[
        ModelPrecisionCompileSummary, CompileScorecardJob, ScorecardCompilePath
    ]
):
    model_summary_type = ModelPrecisionCompileSummary
    scorecard_job_type = CompileScorecardJob

    def get_run(
        self,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardCompilePath | ScorecardProfilePath,
        component: str | None = None,
        universal_device_fallback: bool = True,
    ) -> CompileScorecardJob:
        if isinstance(path, ScorecardProfilePath):
            path = path.compile_path

        if model_summary := self.summaries_per_precision.get(precision):
            return model_summary.get_run(
                device, path, component, universal_device_fallback
            )

        # Create a "Skipped" run to return
        return self.__class__.scorecard_job_type(
            self.model_id, precision, None, device, False, None, path
        )


# --------------------------------------
#
# Inference Job Summary Classes
#


class DeviceInferenceSummary(
    ScorecardDeviceSummary[InferenceScorecardJob, ScorecardProfilePath]
):
    scorecard_job_type = InferenceScorecardJob


class ModelPrecisionInferenceSummary(
    ScorecardModelPrecisionSummary[
        DeviceInferenceSummary, InferenceScorecardJob, ScorecardProfilePath
    ]
):
    device_summary_type = DeviceInferenceSummary
    scorecard_job_type = InferenceScorecardJob


class ModelInferenceSummary(
    ScorecardModelSummary[
        ModelPrecisionInferenceSummary, InferenceScorecardJob, ScorecardProfilePath
    ]
):
    model_summary_type = ModelPrecisionInferenceSummary
    scorecard_job_type = InferenceScorecardJob
