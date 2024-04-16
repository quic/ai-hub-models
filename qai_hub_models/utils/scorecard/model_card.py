# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import datetime
import functools
import multiprocessing
import pprint
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union

import qai_hub as hub

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.utils.config_loaders import MODEL_IDS
from qai_hub_models.utils.scorecard.common import (
    REFERENCE_DEVICE_PER_SUPPORTED_CHIPSETS,
)
from qai_hub_models.utils.scorecard.job_summary import (
    CompileJobSummary,
    ProfileJobSummary,
)


def supported_chipsets(chips: List[str]) -> List[str]:
    """Return all the supported chipsets given the chipset it works on."""

    # Don't assign "chips" directly to supported_chips.
    # The lists will share the same pointer, and hence the for
    # loop below will break.
    supported_chips = set(chips)

    for chip in chips:
        if chip == "qualcomm-snapdragon-8gen3":
            supported_chips.update(
                [
                    "qualcomm-snapdragon-8gen2",
                    "qualcomm-snapdragon-8gen1",
                    "qualcomm-snapdragon-888",
                ]
            )
        if chip == "qualcomm-snapdragon-8gen2":
            supported_chips.update(
                [
                    "qualcomm-snapdragon-8gen3",
                    "qualcomm-snapdragon-8gen1",
                    "qualcomm-snapdragon-888",
                ]
            )
        if chip == "qualcomm-snapdragon-855":
            supported_chips.update(
                ["qualcomm-snapdragon-845", "qualcomm-snapdragon-865"]
            )

    return sorted(list(supported_chips))


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
    return sorted(
        list(set([chipset_marketing_name(chip) for chip in supported_chipsets(chips)]))
    )


# Caching this information is helpful because it requires pulling data from hub.
# Pulling data from hub is slow.
__CHIP_SUPPORTED_DEVICES_CACHE: Dict[str, List[str]] = {}


def supported_devices(chips) -> List[str]:
    """Return all the supported devices given the chipset being used."""
    supported_devices = set(
        [
            "Google Pixel 3",
            "Google Pixel 3a",
            "Google Pixel 4",
            "Google Pixel 3a XL",
            "Google Pixel 4a",
            "Google Pixel 5a 5G",
        ]
    )

    for chip in supported_chipsets(chips):
        supported_devices_for_chip = __CHIP_SUPPORTED_DEVICES_CACHE.get(chip, list())
        if not supported_devices_for_chip:
            supported_devices_for_chip = [
                device.name for device in hub.get_devices(attributes=f"chipset:{chip}")
            ]
            __CHIP_SUPPORTED_DEVICES_CACHE[chip] = supported_devices_for_chip
        supported_devices.update(supported_devices_for_chip)

    return sorted(list(supported_devices))


def supported_oses() -> List[str]:
    """Return all the supported operating systems."""
    return ["Android"]


# Caching this information is helpful because it requires pulling data from hub.
# Pulling data from hub is slow.
__REFERENCE_DEVICE_INFO_PER_CHIPSET = {}


def get_reference_device_info(chipset: str) -> Dict[str, str]:
    if chipset not in __REFERENCE_DEVICE_INFO_PER_CHIPSET:
        hub_device = REFERENCE_DEVICE_PER_SUPPORTED_CHIPSETS[chipset]
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
class ChipsetPerfSummary:
    chipset_name: str
    run_per_runtime: Dict[TargetRuntime, ProfileJobSummary]  # Map<Runtime, Summary>

    @staticmethod
    def from_runs(chipset_name: str, runtime_runs: List[ProfileJobSummary]):
        # Figure out unique devices in various baselines
        run_per_runtime: Dict[TargetRuntime, ProfileJobSummary] = {}
        for run in runtime_runs:
            assert run.chipset == chipset_name  # Chipset should match
            run_per_runtime[run.runtime] = run

        return ChipsetPerfSummary(chipset_name, run_per_runtime)

    def get_perf_card(self) -> Dict[str, str | Dict[str, str]]:
        perf_card: Dict[str, str | Dict[str, str]] = {}
        for runtime, run in self.run_per_runtime.items():
            if not run.skipped:  # Skipped runs are not included
                perf_card[runtime.long_name] = run.performance_metrics
        perf_card["reference_device_info"] = get_reference_device_info(
            self.chipset_name
        )
        perf_card["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
        return perf_card

    def __repr__(self) -> str:
        return pprint.pformat(self.get_perf_card())


@dataclass
class ModelPerfSummary:
    model_id: str
    runs_per_chipset: Dict[str, ChipsetPerfSummary]  # Map<Device Name, Summary>

    @staticmethod
    def from_runs(model_id: str, device_runs: List[ProfileJobSummary]):
        # Figure out unique devices in various baselines
        runs_per_chipset: Dict[str, List[ProfileJobSummary]] = {}
        for run in device_runs:
            assert run.model_id == model_id  # All should have the same model ID
            list = runs_per_chipset.get(run.chipset or "", [])
            runs_per_chipset[run.chipset] = list
            list.append(run)

        return ModelPerfSummary(
            model_id,
            {
                chipset_name: ChipsetPerfSummary.from_runs(chipset_name, runs)
                for chipset_name, runs in runs_per_chipset.items()
            },
        )

    def get_perf_card(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        perf_card = []
        for summary in self.runs_per_chipset.values():
            perf_card.append(summary.get_perf_card())
        return perf_card

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


@dataclass
class PerfSummary:
    runs_per_model: Dict[str, ModelPerfSummary]  # Map<Model ID, Summary>

    @staticmethod
    def from_model_ids(
        job_ids: Dict[str, str], model_ids=MODEL_IDS
    ) -> Dict[str, PerfSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>|<runtime>|<device>|<model_component_id> : job_id
            <model_id>|<runtime>|<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print("Generating Performance Summary for Models")
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.map(
            functools.partial(PerfSummary.from_model_id, job_ids=job_ids), model_ids
        )
        pool.close()
        print("Finished\n")
        return {k: v for k, v in model_summaries}

    @staticmethod
    def from_model_id(
        model_id: str, job_ids: Dict[str, str]
    ) -> Tuple[str, PerfSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>|<runtime>|<device>|<model_component_id> : job_id
            <model_id>|<runtime>|<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print(f"    {model_id} ")
        runs = ProfileJobSummary.from_model_id(model_id, job_ids)
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
        for _, model_summary in self.runs_per_model.items():
            chips.update(model_summary.runs_per_chipset.keys())
        return chips

    def get_perf_card(self) -> Dict[str, str | List[Any] | Dict[str, Any]]:
        perf_card: Dict[str, str | List[Any] | Dict[str, Any]] = {}

        chips = self.get_chipsets()
        perf_card["aggregated"] = dict(
            supported_oses=supported_oses(),
            supported_devices=supported_devices(chips),
            supported_chipsets=supported_chipsets_santized(chips),
        )

        models_list: List[Dict[str, Any]] = []
        for model_id, summary in self.runs_per_model.items():
            models_list.append(
                {"name": model_id, "performance_metrics": summary.get_perf_card()}
            )
        perf_card["models"] = models_list
        return perf_card

    def __repr__(self):
        return pprint.pformat(self.get_perf_card())


@dataclass
class ModelCompileSummary:
    model_id: str
    runs_per_runtime: Dict[
        TargetRuntime, CompileJobSummary
    ]  # Map<Device Name, Summary>

    @staticmethod
    def from_runs(model_id: str, runtime_runs: List[CompileJobSummary]):
        run_per_runtime: Dict[TargetRuntime, CompileJobSummary] = {}
        for run in runtime_runs:
            assert run.model_id == model_id  # model id should match
            run_per_runtime[run.runtime] = run
        return ModelCompileSummary(model_id, run_per_runtime)


@dataclass
class CompileSummary:
    runs_per_model: Dict[str, ModelCompileSummary]  # Map<Model ID, Summary>

    @staticmethod
    def from_model_ids(
        job_ids: Dict[str, str], model_ids=MODEL_IDS
    ) -> Dict[str, CompileSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>|<runtime>|<device>|<model_component_id> : job_id
            <model_id>|<runtime>|<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print("Generating Compilation Summary for Models")
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.map(
            functools.partial(CompileSummary.from_model_id, job_ids=job_ids), model_ids
        )
        pool.close()
        print("Finished\n")
        return {k: v for k, v in model_summaries}

    @staticmethod
    def from_model_id(
        model_id: str, job_ids: Dict[str, str]
    ) -> Tuple[str, CompileSummary]:
        """
        Reads jobs for every `model_id` from the dictionary and creates summaries for each. `job_ids` format:
        Either:
            <model_id>|<runtime>|<device>|<model_component_id> : job_id
            <model_id>|<runtime>|<device> : job_id

        Returns models in this format:
            model_id: List[Summary]
        """
        print(f"    {model_id} ")
        runs = CompileJobSummary.from_model_id(model_id, job_ids)
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
