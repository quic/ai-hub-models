# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import multiprocessing
import pprint
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

import qai_hub as hub

from qai_hub_models.utils.config_loaders import MODEL_IDS
from qai_hub_models.utils.scorecard.common import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.utils.scorecard.job_summary import (
    CompileJobSummary,
    ProfileJobSummary,
)


def supported_chipsets(chips: List[str]) -> List[str]:
    """
    Return all the supported chipsets given the chipset it works on.

    The order of chips in the website list mirror the order here. Order
    chips from newest to oldest to highlight newest chips most prominently.
    """
    chipset_set = set(chips)
    chipset_list = []
    if "qualcomm-snapdragon-8gen3" in chipset_set:
        chipset_list.extend(
            [
                "qualcomm-snapdragon-8gen3",
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
        )
    elif "qualcomm-snapdragon-8gen2" in chipset_set:
        chipset_list.extend(
            [
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
        )

    chipset_order = [
        "qualcomm-snapdragon-x-elite",
        "qualcomm-qcs6490",
        "qualcomm-qcs8250",
        "qualcomm-qcs8550",
        "qualcomm-sa8775p",
        "qualcomm-sa8650p",
        "qualcomm-sa8255p",
    ]
    for chipset in chipset_order:
        if chipset in chipset_set:
            chipset_list.append(chipset)

    # Add any remaining chipsets not covered
    for chipset in chipset_set:
        if chipset not in chipset_list:
            chipset_list.append(chipset)
    return chipset_list


def chipset_marketing_name(chipset) -> str:
    """Sanitize chip name to match marketting."""
    chip = [word.capitalize() for word in chipset.split("-")]
    details_to_remove = []
    for i in range(len(chip)):
        if chip[i] == "8gen3":
            chip[i] = "8 Gen 3"
        if chip[i] == "8gen2":
            chip[i] = "8 Gen 2"
        elif chip[i] == "8gen1":
            chip[i] = "8 Gen 1"
        elif chip[i] == "Snapdragon":
            # Marketing name for Qualcomm Snapdragon is Snapdragon®
            chip[i] = "Snapdragon®"
        elif chip[i] == "Qualcomm":
            details_to_remove.append(chip[i])

    for detail in details_to_remove:
        chip.remove(detail)
    return " ".join(chip)


def supported_chipsets_santized(chips) -> List[str]:
    """Santize the chip name passed via hub."""
    chips = [chip for chip in chips if chip != ""]
    return [chipset_marketing_name(chip) for chip in supported_chipsets(chips)]


# Caching this information is helpful because it requires pulling data from hub.
# Pulling data from hub is slow.
__CHIP_SUPPORTED_DEVICES_CACHE: Dict[str, List[str]] = {}


def get_supported_devices(chips) -> List[str]:
    """Return all the supported devices given the chipset being used."""
    supported_devices = []

    for chip in supported_chipsets(chips):
        supported_devices_for_chip = __CHIP_SUPPORTED_DEVICES_CACHE.get(chip, list())
        if not supported_devices_for_chip:
            supported_devices_for_chip = [
                device.name
                for device in hub.get_devices(attributes=f"chipset:{chip}")
                if "(Family)" not in device.name
            ]
            supported_devices_for_chip = sorted(set(supported_devices_for_chip))
            __CHIP_SUPPORTED_DEVICES_CACHE[chip] = supported_devices_for_chip
        supported_devices.extend(supported_devices_for_chip)
    supported_devices.extend(
        [
            "Google Pixel 5a 5G",
            "Google Pixel 4",
            "Google Pixel 4a",
            "Google Pixel 3",
            "Google Pixel 3a",
            "Google Pixel 3a XL",
        ]
    )
    return supported_devices


def supported_oses() -> List[str]:
    """Return all the supported operating systems."""
    return ["Android"]


# Caching this information is helpful because it requires pulling data from hub.
# Pulling data from hub is slow.
__REFERENCE_DEVICE_INFO_PER_CHIPSET = {}


def get_reference_device_info(device: ScorecardDevice) -> Dict[str, str]:
    chipset = device.get_chipset()
    if chipset not in __REFERENCE_DEVICE_INFO_PER_CHIPSET:
        hub_device = device.get_reference_device()
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


@dataclass
class DevicePerfSummary:
    device: ScorecardDevice
    run_per_path: Dict[ScorecardProfilePath, ProfileJobSummary]  # Map<path, Summary>

    @staticmethod
    def from_runs(device: ScorecardDevice, path_runs: List[ProfileJobSummary]):
        # Figure out unique devices in various baselines
        run_per_path: Dict[ScorecardProfilePath, ProfileJobSummary] = {}
        for run in path_runs:
            assert run._device == device  # Device should match
            run_per_path[run.path] = run

        return DevicePerfSummary(device, run_per_path)

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
    ) -> Dict[str, str | Dict[str, str]]:
        perf_card: Dict[str, str | Dict[str, str]] = {}
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


@dataclass
class ModelPerfSummary:
    model_id: str
    runs_per_device: Dict[
        ScorecardDevice, DevicePerfSummary
    ]  # Map<Device Name, Summary>

    @staticmethod
    def from_runs(model_id: str, device_runs: List[ProfileJobSummary]):
        # Figure out unique devices in various baselines
        runs_per_device: Dict[ScorecardDevice, List[ProfileJobSummary]] = {}
        for run in device_runs:
            assert run.model_id == model_id  # All should have the same model ID
            list = runs_per_device.get(run._device, [])
            runs_per_device[run._device] = list
            list.append(run)

        return ModelPerfSummary(
            model_id,
            {
                device: DevicePerfSummary.from_runs(device, runs)
                for device, runs in runs_per_device.items()
            },
        )

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
    ) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        perf_card = []
        for summary in self.runs_per_device.values():
            device_summary = summary.get_perf_card(include_failed_jobs, exclude_paths)
            # If device had no runs, omit it from the card
            if device_summary:
                perf_card.append(device_summary)
        return perf_card

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


@dataclass
class PerfSummary:
    runs_per_model: Dict[str, ModelPerfSummary]  # Map<Model ID, Summary>

    @staticmethod
    def from_model_ids(
        job_ids: Dict[str, str],
        model_ids=MODEL_IDS,
        max_job_wait_secs: int | None = None,
    ) -> Dict[str, PerfSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>_<path>-<device>_<model_component_id> : job_id
            <model_id>_<path>-<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print("Generating Performance Summary for Models")
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.map(
            functools.partial(
                PerfSummary.from_model_id,
                job_ids=job_ids,
                max_job_wait_secs=max_job_wait_secs,
            ),
            model_ids,
        )
        pool.close()
        print("Finished\n")
        return {k: v for k, v in model_summaries}

    @staticmethod
    def from_model_id(
        model_id: str,
        job_ids: Dict[str, str],
        max_job_wait_secs: int | None = None,
    ) -> Tuple[str, PerfSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>_<path>-<device>_<model_component_id> : job_id
            <model_id>_<path>-<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print(f"    {model_id} ")
        runs = ProfileJobSummary.from_model_id(model_id, job_ids, max_job_wait_secs)
        return model_id, PerfSummary.from_runs(runs)

    @staticmethod
    def from_runs(model_runs: List[ProfileJobSummary]):
        # Figure out unique models in various baselines
        runs_per_model: Dict[str, List[ProfileJobSummary]] = {}
        for run in model_runs:
            list = runs_per_model.get(run.model_id, [])
            list.append(run)
            runs_per_model[run.model_id] = list

        return PerfSummary(
            {
                model_id: ModelPerfSummary.from_runs(model_id, runs)
                for model_id, runs in runs_per_model.items()
            }
        )

    def get_chipsets(self) -> Set[str]:
        chips: Set[str] = set()
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
                if model_id in device.get_disabled_models():
                    continue

                chips.add(device.get_chipset())
        return chips

    def get_perf_card(
        self,
        include_failed_jobs: bool = True,
        exclude_paths: Iterable[ScorecardProfilePath] = [],
    ) -> Dict[str, str | List[Any] | Dict[str, Any]]:
        perf_card: Dict[str, str | List[Any] | Dict[str, Any]] = {}

        chips = self.get_chipsets()
        perf_card["aggregated"] = dict(
            supported_oses=supported_oses(),
            supported_devices=get_supported_devices(chips),
            supported_chipsets=supported_chipsets_santized(chips),
        )

        models_list: List[Dict[str, Any]] = []
        for model_id, summary in self.runs_per_model.items():
            models_list.append(
                {
                    "name": model_id,
                    "performance_metrics": summary.get_perf_card(
                        include_failed_jobs, exclude_paths
                    ),
                }
            )
        perf_card["models"] = models_list
        return perf_card

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


@dataclass
class DeviceCompileSummary:
    device: ScorecardDevice
    run_per_path: Dict[ScorecardCompilePath, CompileJobSummary]  # Map<path, Summary>

    @staticmethod
    def from_runs(device: ScorecardDevice, path_runs: List[CompileJobSummary]):
        # Figure out unique devices in various baselines
        run_per_path: Dict[ScorecardCompilePath, CompileJobSummary] = {}
        for run in path_runs:
            assert run._device == device  # Device should match
            run_per_path[run.path] = run

        return DeviceCompileSummary(device, run_per_path)


@dataclass
class ModelCompileSummary:
    model_id: str
    runs_per_device: Dict[
        ScorecardDevice, DeviceCompileSummary
    ]  # Map<Device Name, Summary>

    @staticmethod
    def from_runs(model_id: str, path_runs: List[CompileJobSummary]):
        runs_per_device: Dict[ScorecardDevice, List[CompileJobSummary]] = {}
        for run in path_runs:
            assert run.model_id == model_id  # model id should match
            list = runs_per_device.get(run._device, [])
            runs_per_device[run._device] = list
            list.append(run)
        return ModelCompileSummary(
            model_id,
            {
                device: DeviceCompileSummary.from_runs(device, runs)
                for device, runs in runs_per_device.items()
            },
        )


@dataclass
class CompileSummary:
    runs_per_model: Dict[str, ModelCompileSummary]  # Map<Model ID, Summary>

    @staticmethod
    def from_model_ids(
        job_ids: Dict[str, str],
        model_ids=MODEL_IDS,
        max_job_wait_secs: int | None = None,
    ) -> Dict[str, CompileSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>_<runtime>-<device>_<model_component_id> : job_id
            <model_id>_<runtime>-<device> : job_id
            <model_id>_<runtime>_<model_component_id> : job_id
            <model_id>_<runtime> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print("Generating Compilation Summary for Models")
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.map(
            functools.partial(
                CompileSummary.from_model_id,
                job_ids=job_ids,
                max_job_wait_secs=max_job_wait_secs,
            ),
            model_ids,
        )
        pool.close()
        print("Finished\n")
        return {k: v for k, v in model_summaries}

    @staticmethod
    def from_model_id(
        model_id: str,
        job_ids: Dict[str, str],
        max_job_wait_secs: int | None = None,
    ) -> Tuple[str, CompileSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>_<runtime>-<device>_<model_component_id> : job_id
            <model_id>_<runtime>-<device> : job_id
            <model_id>_<runtime>_<model_component_id> : job_id
            <model_id>_<runtime> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print(f"    {model_id} ")
        runs = CompileJobSummary.from_model_id(model_id, job_ids, max_job_wait_secs)
        return model_id, CompileSummary.from_runs(runs)

    @staticmethod
    def from_runs(model_runs: List[CompileJobSummary]) -> "CompileSummary":
        # Figure out unique models in various baselines
        runs_per_model: Dict[str, List[CompileJobSummary]] = {}
        for run in model_runs:
            list = runs_per_model.get(run.model_id, [])
            list.append(run)
            runs_per_model[run.model_id] = list

        return CompileSummary(
            {
                model_id: ModelCompileSummary.from_runs(model_id, runs)
                for model_id, runs in runs_per_model.items()
            }
        )
