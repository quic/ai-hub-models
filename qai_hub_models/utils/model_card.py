# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import qai_hub as hub


def chipset_marketting_name(chipset) -> str:
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


class MODEL_CARD_RUNTIMES(Enum):
    """Runtime to be stored in model card."""

    TORCHSCRIPT_ONNX_TFLITE = 100
    TORCHSCRIPT_ONNX_QNN = 101

    @staticmethod
    def from_string(string: str) -> "MODEL_CARD_RUNTIMES":
        return MODEL_CARD_RUNTIMES["TORCHSCRIPT_ONNX_" + string.upper()]


@dataclass
class ModelRun:
    model_id: str
    profile_job_id: str
    runtime: MODEL_CARD_RUNTIMES
    device_type: str

    def chipset(self) -> Optional[str]:
        """Chipset the job was run on."""
        if self.profile_job is not None:
            hub_device = self.profile_job.device
            for attr in hub_device.attributes:
                if attr.startswith("chipset:qualcomm"):
                    return attr.split(":")[1]
        return ""

    @property
    def profile_job(self):
        """Get the hub.ProfileJob object."""
        if len(self.profile_job_id) > 0:
            job = hub.get_job(self.profile_job_id)
            job.wait()
            return job
        return None

    def job_status(self) -> str:
        """Get the job status of the profile job."""
        if self.profile_job is not None:
            if self.profile_job.get_status().success:
                return "Passed"
            elif self.profile_job.get_status().failure:
                return "Failed"
        return "Skipped"

    @property
    def quantized(self) -> str:
        """Quantized models are marked so precision can be correctly recorded."""
        return (
            "Yes"
            if self.model_id.endswith("Quantized")
            or self.model_id.endswith("Quantizable")
            else "No"
        )

    @property
    def profile_results(self):
        """Profile results from profile job."""
        if self.job_status() == "Passed":
            return self.profile_job.download_profile()
        return None

    def get_inference_time(self) -> Union[float, str]:
        """Get the inference time from the profile job."""
        if self.profile_results is not None:
            return float(
                self.profile_results["execution_summary"]["estimated_inference_time"]
            )
        return "null"

    def get_throughput(self) -> Union[float, str]:
        """Get the throughput from the profile job."""
        if not isinstance(self.get_inference_time(), str):
            return 1000000 / self.get_inference_time()  # type: ignore
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

    def npu(self) -> Any:
        """Get number of layers running on NPU."""
        return self.get_layer_info("NPU") if self.profile_results is not None else 0

    def gpu(self) -> Any:
        """Get number of layers running on GPU."""
        return self.get_layer_info("GPU") if self.profile_results is not None else 0

    def cpu(self) -> Any:
        """Get number of layers running on CPU."""
        return self.get_layer_info("CPU") if self.profile_results is not None else 0

    def total(self) -> Any:
        """Get the total number of layers."""
        return self.npu() + self.gpu() + self.cpu()

    def primary_compute_unit(self) -> str:
        """Get the primary compute unit."""
        layers_npu = self.npu()
        layers_gpu = self.gpu()
        layers_cpu = self.cpu()

        if layers_npu == 0 and layers_gpu == 0 and layers_cpu == 0:
            return "null"
        compute_unit_for_most_layers = max(layers_cpu, layers_gpu, layers_npu)
        if compute_unit_for_most_layers == layers_npu:
            return "NPU"
        elif compute_unit_for_most_layers == layers_gpu:
            return "GPU"
        return "CPU"

    def get_peak_memory_range(self) -> Dict[str, int]:
        """Get the estimated peak memory range."""
        if self.profile_results is not None:
            low, high = self.profile_results["execution_summary"][
                "inference_memory_peak_range"
            ]
            return dict(min=low, max=high)
        return dict(min=0, max=0)

    def precision(self) -> str:
        """Get the precision of the model based on the run."""
        if self.profile_results is not None:
            compute_unit = self.primary_compute_unit()
            if compute_unit == "CPU":
                return "fp32"
            if self.quantized == "Yes":
                return "int8"
            return "fp16"
        return "null"

    def performance_metrics(self) -> Dict[str, Any]:
        return dict(
            inference_time=self.get_inference_time(),
            throughput=self.get_throughput(),
            estimated_peak_memory_range=self.get_peak_memory_range(),
            primary_compute_unit=self.primary_compute_unit(),
            precision=self.precision(),
            layer_info=dict(
                layers_on_npu=self.npu(),
                layers_on_gpu=self.gpu(),
                layers_on_cpu=self.cpu(),
                total_layers=self.total(),
            ),
            job_id=self.profile_job_id,
            job_status=self.job_status(),
        )

    def reference_device_info(self) -> Dict[str, str]:
        """Return a reference ID."""
        REF_DEVICE_MAP = {
            "s23": ("qualcomm-snapdragon-8gen2", "Samsung Galaxy S23"),
            "s24": ("qualcomm-snapdragon-8gen3", "Samsung Galaxy S24"),
        }
        chipset = REF_DEVICE_MAP[self.device_type][0]
        hub_device = hub.get_devices(REF_DEVICE_MAP[self.device_type][1])[0]
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
        chipset = chipset_marketting_name(chipset)
        device_info = dict(
            name=device_name,
            os=os_version,
            form_factor=form_factor.capitalize(),
            os_name=os_name,
            manufacturer=manufacturer.capitalize(),
            chipset=chipset,
        )
        return device_info


@dataclass
class ModelPerf:
    model_runs: List[ModelRun]

    def supported_chipsets(self, chips: List[str]) -> List[str]:
        """Return all the supported chipsets given the chipset it works on."""

        # Don't assign "chips" directly to supported_chips.
        # The lists will share the same pointer, and hence the for
        # loop below will break.
        supported_chips = []
        supported_chips.extend(chips)

        for chip in chips:
            if chip == "qualcomm-snapdragon-8gen3":
                supported_chips.extend(
                    [
                        "qualcomm-snapdragon-8gen2",
                        "qualcomm-snapdragon-8gen1",
                        "qualcomm-snapdragon-888",
                    ]
                )
            if chip == "qualcomm-snapdragon-8gen2":
                supported_chips.extend(
                    [
                        "qualcomm-snapdragon-8gen3",
                        "qualcomm-snapdragon-8gen1",
                        "qualcomm-snapdragon-888",
                    ]
                )
            if chip == "qualcomm-snapdragon-855":
                supported_chips.extend(
                    ["qualcomm-snapdragon-845", "qualcomm-snapdragon-865"]
                )
        return supported_chips

    def supported_chipsets_santized(self, chips) -> List[str]:
        """Santize the chip name passed via hub."""
        chips = [chip for chip in chips if chip != ""]
        return sorted(
            list(
                set(
                    [
                        chipset_marketting_name(chip)
                        for chip in self.supported_chipsets(chips)
                    ]
                )
            )
        )

    def supported_devices(self, chips) -> List[str]:
        """Return all the supported devicesgiven the chipset being used."""
        supported_devices = []
        for chip in self.supported_chipsets(chips):
            supported_devices.extend(
                [
                    device.name
                    for device in hub.get_devices(attributes=f"chipset:{chip}")
                ]
            )
        supported_devices.extend(
            [
                "Google Pixel 3",
                "Google Pixel 3a",
                "Google Pixel 4",
                "Google Pixel 3a XL",
                "Google Pixel 4a",
                "Google Pixel 5a 5G",
            ]
        )
        return sorted(list(set(supported_devices)))

    def supported_oses(self) -> List[str]:
        """Return all the supported operating systems."""
        return ["Android"]

    def performance_metrics(self):
        """Performance metrics as per model card."""
        perf_card = dict()

        # Figure out unique models in various baselines
        unique_model_ids = []
        chips = []
        devices = []
        for run in self.model_runs:
            if run.model_id not in unique_model_ids:
                unique_model_ids.append(run.model_id)
            if run.chipset not in chips:
                chips.append(run.chipset())
            if run.device_type not in devices:
                devices.append(run.device_type)

        perf_card["aggregated"] = dict(
            supported_oses=self.supported_oses(),
            supported_devices=self.supported_devices(chips),
            supported_chipsets=self.supported_chipsets_santized(chips),
        )

        perf_per_model = []

        for mid in unique_model_ids:
            # Calculate per data per runtime
            perf_per_device = dict()
            for run in self.model_runs:
                if run.model_id == mid:
                    for dev in devices:
                        if run.device_type == dev:
                            # perf_per_runtime = dict()
                            if dev not in perf_per_device:
                                perf_per_device[dev] = dict()
                            runtime_name = run.runtime.name.lower()
                            perf_per_device[dev][
                                runtime_name
                            ] = run.performance_metrics()
                            # Per model, the device used and timestamp for model card
                            if "reference_device_info" not in perf_per_device[dev]:
                                perf_per_device[dev][
                                    "reference_device_info"
                                ] = run.reference_device_info()

                            perf_per_device[dev]["timestamp"] = (
                                datetime.datetime.utcnow().isoformat() + "Z"
                            )

                perf_model = dict(
                    name=mid, performance_metrics=list(perf_per_device.values())
                )
            perf_per_model.append(perf_model)

        # Perf card with multiple models
        perf_card["models"] = perf_per_model
        return perf_card
