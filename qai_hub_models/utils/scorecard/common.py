# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from enum import Enum
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import qai_hub as hub

from qai_hub_models.models.common import TargetRuntime

_DEVICE_CACHE: Dict[str, hub.Device] = {}


def _get_cached_device(device_name: str) -> hub.Device:
    # Gets a device with attributes & OS. This only comes from hub.get_devices()
    device = _DEVICE_CACHE.get(device_name, None)
    if not device:
        device = hub.get_devices(device_name)[0]
        _DEVICE_CACHE[device_name] = device
    return device


def scorecard_unit_test_idfn(val):
    """Name of unit test parameters used in tests created in test_generated.py"""
    if val == ScorecardDevices.any:
        return "device_agnostic"
    elif isinstance(val, ScorecardDevice):
        return val.name


class ScorecardDevice:
    # -- DEVICE REGISTRY --
    _registry: Dict[str, "ScorecardDevice"] = {}

    @classmethod
    def all_enabled(cls) -> List["ScorecardDevice"]:
        return [x for x in cls._registry.values() if x.enabled]

    @classmethod
    def register(
        cls,
        name: str,
        reference_device_name: Optional[str],
        execution_device_name: Optional[str] = None,
        disabled_models: List[str] = [],
    ) -> "ScorecardDevice":
        if name in cls._registry:
            raise ValueError("Device " + name + "already registered.")

        device = ScorecardDevice(
            name, reference_device_name, execution_device_name, disabled_models
        )
        cls._registry[name] = device
        return device

    @classmethod
    def __getitem__(cls, device_name: str) -> "ScorecardDevice":
        return cls._registry[device_name]

    # -- DEVICE CLASS --
    def __init__(
        self,
        name: str,
        reference_device_name: Optional[str],
        execution_device_name: Optional[str] = None,
        disabled_models: List[str] = [],
    ):
        """
        Parameters
            name: Name of this device for scorecard use. Must conform to the name of a python enum.

            reference_device_name: The name of the "reference" device used by the scorecard for metadata when coalating results.

            execution_device_name: The name of the device to be used by associated Hub jobs.
                                   If not provided, jobs will be submitted with the chipset of the reference device.
                                   Hub will decide what device to use depending on availability.

            disabled_models: AI Hub Model IDs that are not supported by this device.
                            These models will be ignored by the scorecard in combination with this device.
        """
        self.name = name
        self.disabled_models = disabled_models
        self.reference_device_name = reference_device_name
        self.execution_device_name = execution_device_name

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.name.lower()

    def __eq__(self, other):
        if not isinstance(other, ScorecardDevice):
            return False
        return (
            self.name == other.name
            and self.reference_device_name == other.reference_device_name
            and self.execution_device_name == other.execution_device_name
        )

    def __hash__(self):
        return (
            hash(self.name)
            + hash(self.reference_device_name)
            + hash(self.execution_device_name)
        )

    @property
    def enabled(self) -> bool:
        """
        Whether the scorecard should include this scorecard device.
        This applies both to submitted jobs and analyses applied to an existing scorecard job yaml.
        """
        valid_test_devices = os.environ.get("WHITELISTED_PROFILE_TEST_DEVICES", "ALL")
        return (
            valid_test_devices == "ALL"
            or self.name == "all"
            or self.name in valid_test_devices.split(",")
        )

    @property
    def public(self) -> bool:
        return self in ScorecardDevices.__dict__.values()

    @cached_property
    def reference_device(self) -> hub.Device:
        """
        Get the "reference" device used by the scorecard for metadata when coalating results.
        This is not used by any actual scorecard jobs.
        """
        if self.reference_device_name:
            return _get_cached_device(self.reference_device_name)
        raise NotImplementedError(f"No reference device for {self.name}")

    @cached_property
    def execution_device(self) -> hub.Device:
        """
        Get the "reference" device used by the scorecard for metadata when coalating results.
        This is not used by any actual scorecard jobs.
        """
        if self.execution_device_name:
            return _get_cached_device(self.execution_device_name)
        raise NotImplementedError(f"No execution device for {self.name}")

    @cached_property
    def chipset(self) -> str:
        """
        The chipset used by this device.
        """
        device = (
            self.execution_device
            if self.execution_device_name
            else self.reference_device
        )
        for attr in device.attributes:
            if attr.startswith("chipset:"):
                return attr[8:]
        raise ValueError(f"Chipset not found for device: {self.name}")

    @cached_property
    def os(self) -> str:
        """
        The operating system used by this device.
        """
        for attr in self.reference_device.attributes:
            if attr.startswith("os:"):
                return attr[3:]
        raise ValueError(f"OS Not found for device: {self.name}")


class ScorecardDevices:
    any = ScorecardDevice.register(
        "any", "Samsung Galaxy S23"
    )  # no specific device (usable only during compilation)
    cs_8_gen_2 = ScorecardDevice.register("cs_8_gen_2", "Samsung Galaxy S23")
    cs_8_gen_3 = ScorecardDevice.register(
        "cs_8_gen_3", "Samsung Galaxy S24", "Samsung Galaxy S24 (Family)"
    )
    cs_6490 = ScorecardDevice.register(
        "cs_6490",
        "RB3 Gen 2 (Proxy)",
        None,
        [
            "ConvNext-Tiny-w8a8-Quantized",
            "ConvNext-Tiny-w8a16-Quantized",
            "ResNet50Quantized",
            "RegNetQuantized",
            "HRNetPoseQuantized",
            "SESR-M5-Quantized",
            "Midas-V2-Quantized",
            "Posenet-Mobilenet-Quantized",
        ],
    )
    cs_8250 = ScorecardDevice.register("cs_8250", "RB5 (Proxy)")
    cs_8550 = ScorecardDevice.register("cs_8550", "QCS8550 (Proxy)")
    cs_x_elite = ScorecardDevice.register("cs_x_elite", "Snapdragon X Elite CRD")
    cs_auto_lemans_8255 = ScorecardDevice.register(
        "cs_auto_lemans_8255", "SA8255 (Proxy)"
    )
    cs_auto_lemans_8775 = ScorecardDevice.register(
        "cs_auto_lemans_8775", "SA8775 (Proxy)"
    )
    cs_auto_lemans_8650 = ScorecardDevice.register(
        "cs_auto_lemans_8650", "SA8650 (Proxy)"
    )
    cs_xr_8450 = ScorecardDevice.register("cs_xr_8450", "QCS8450 (Proxy)")
    cs_auto_makena_8295 = ScorecardDevice.register(
        "cs_auto_makena_8295", "Snapdragon Cockpit Gen 4 QAM"
    )


def get_job_cache_name(
    runtime_name: str,
    model_name: str,
    device: ScorecardDevice,
    component: Optional[str] = None,
) -> str:
    return (
        f"{model_name}_{runtime_name}"
        + ("-" + device.name if device != ScorecardDevices.any else "")
        + ("_" + component if component else "")
    )


class ScorecardCompilePath(Enum):
    TFLITE = 0
    QNN = 1
    ONNX = 2
    ONNX_FP16 = 3

    def __str__(self):
        return self.name.lower()

    @property
    def long_name(self):
        if "onnx" in self.name.lower():
            return f"torchscript_{self.name.lower()}"
        return f"torchscript_onnx_{self.name.lower()}"

    def enabled(self) -> bool:
        valid_test_runtimes = os.environ.get("WHITELISTED_TEST_RUNTIMES", "ALL")
        return valid_test_runtimes == "ALL" or (
            self.get_runtime().name.lower()
            in [x.lower() for x in valid_test_runtimes.split(",")]
        )

    @staticmethod
    def all_enabled() -> List["ScorecardCompilePath"]:
        return [x for x in ScorecardCompilePath if x.enabled()]

    @staticmethod
    def get_parameterized_test_config(
        aimet_model=False,
        only_enabled_paths=True,
        only_enabled_devices=True,
    ) -> List[Tuple["ScorecardCompilePath", ScorecardDevice]]:
        path_list: List[ScorecardCompilePath] = ScorecardCompilePath.all_enabled() if only_enabled_paths else ScorecardCompilePath  # type: ignore
        path_devices_dict = {
            sc_path: sc_path.get_test_devices(aimet_model, only_enabled_devices)
            for sc_path in path_list
        }
        return [
            (key, dev) for key, devices in path_devices_dict.items() for dev in devices
        ]

    def get_runtime(self) -> TargetRuntime:
        if self == ScorecardCompilePath.TFLITE:
            return TargetRuntime.TFLITE
        if self in [ScorecardCompilePath.ONNX, ScorecardCompilePath.ONNX_FP16]:
            return TargetRuntime.ONNX
        if self == ScorecardCompilePath.QNN:
            return TargetRuntime.QNN
        raise NotImplementedError()

    def get_test_devices(
        self, aimet_model: bool = False, only_enabled: bool = True
    ) -> List[ScorecardDevice]:
        if self == ScorecardCompilePath.QNN:
            devices = [
                ScorecardDevices.any,
                ScorecardDevices.cs_x_elite,
                ScorecardDevices.cs_8550,
                ScorecardDevices.cs_auto_lemans_8255,
                ScorecardDevices.cs_auto_lemans_8775,
                ScorecardDevices.cs_auto_makena_8295,
            ]
            if aimet_model:
                devices.append(ScorecardDevices.cs_6490)
        else:
            devices = [ScorecardDevices.any]

        try:
            from qai_hub_models.utils.scorecard._common_private import (
                get_private_compile_path_test_devices,
            )

            devices.extend(get_private_compile_path_test_devices(self, aimet_model))  # type: ignore
        except ImportError:
            pass

        return [x for x in devices if x.enabled] if only_enabled else devices

    def get_compile_options(self, aimet_model=False) -> str:
        if self == ScorecardCompilePath.ONNX_FP16 and not aimet_model:
            return "--quantize_full_type float16 --quantize_io"
        return ""

    def get_job_cache_name(
        self,
        model: str,
        device: ScorecardDevice = ScorecardDevices.any,
        aimet_model: bool = False,
        component: Optional[str] = None,
    ):
        # These two auto chips are the same, re-use the same compiled asset.
        if device == ScorecardDevices.cs_auto_lemans_8650:
            device = ScorecardDevices.cs_auto_lemans_8775
        if device not in self.get_test_devices(aimet_model=aimet_model):
            device = ScorecardDevices.any  # default to the "generic" compilation path
        return get_job_cache_name(self.name, model, device, component)


class ScorecardProfilePath(Enum):
    TFLITE = 0
    QNN = 1
    ONNX = 2
    ONNX_DML_GPU = 3

    def __str__(self):
        return self.name.lower()

    @property
    def long_name(self):
        if "onnx" in self.name.lower():
            return f"torchscript_{self.name.lower()}"
        return f"torchscript_onnx_{self.name.lower()}"

    def enabled(self) -> bool:
        valid_test_runtimes = os.environ.get("WHITELISTED_TEST_RUNTIMES", "ALL")
        return valid_test_runtimes == "ALL" or (
            self.get_runtime().name.lower()
            in [x.lower() for x in valid_test_runtimes.split(",")]
        )

    @staticmethod
    def all_enabled() -> List["ScorecardProfilePath"]:
        return [x for x in ScorecardProfilePath if x.enabled()]

    def include_in_perf_yaml(self) -> bool:
        return self in [
            ScorecardProfilePath.QNN,
            ScorecardProfilePath.ONNX,
            ScorecardProfilePath.TFLITE,
        ]

    @staticmethod
    def get_parameterized_test_config(
        aimet_model=False,
        only_enabled_paths=True,
        only_enabled_devices=True,
    ) -> List[Tuple["ScorecardProfilePath", ScorecardDevice]]:
        path_list: List[ScorecardProfilePath] = ScorecardProfilePath.all_enabled() if only_enabled_paths else ScorecardProfilePath  # type: ignore
        path_devices_dict = {
            sc_path: sc_path.get_test_devices(aimet_model, only_enabled_devices)
            for sc_path in path_list
        }
        return [
            (key, dev) for key, devices in path_devices_dict.items() for dev in devices
        ]

    def get_runtime(self) -> TargetRuntime:
        if self == ScorecardProfilePath.TFLITE:
            return TargetRuntime.TFLITE
        if self in [ScorecardProfilePath.ONNX, ScorecardProfilePath.ONNX_DML_GPU]:
            return TargetRuntime.ONNX
        if self == ScorecardProfilePath.QNN:
            return TargetRuntime.QNN
        raise NotImplementedError()

    def get_compile_path(self) -> ScorecardCompilePath:
        if self == ScorecardProfilePath.TFLITE:
            return ScorecardCompilePath.TFLITE
        if self == ScorecardProfilePath.ONNX:
            return ScorecardCompilePath.ONNX
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            return ScorecardCompilePath.ONNX_FP16
        if self == ScorecardProfilePath.QNN:
            return ScorecardCompilePath.QNN
        raise NotImplementedError()

    def get_profile_options(self) -> str:
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            return "--compute_unit gpu"
        return ""

    def get_test_devices(
        self, aimet_model: bool = False, only_enabled: bool = True
    ) -> List[ScorecardDevice]:
        if self == ScorecardProfilePath.TFLITE:
            devices = [
                ScorecardDevices.cs_8_gen_2,
                ScorecardDevices.cs_8_gen_3,
                ScorecardDevices.cs_8550,
                ScorecardDevices.cs_xr_8450,
                ScorecardDevices.cs_auto_lemans_8650,
                ScorecardDevices.cs_auto_lemans_8775,
                ScorecardDevices.cs_auto_lemans_8255,
                ScorecardDevices.cs_auto_makena_8295,
            ] + (
                [ScorecardDevices.cs_6490, ScorecardDevices.cs_8250]
                if aimet_model
                else []
            )
        elif self == ScorecardProfilePath.ONNX:
            devices = [
                ScorecardDevices.cs_8_gen_2,
                ScorecardDevices.cs_8_gen_3,
                ScorecardDevices.cs_x_elite,
            ]
        elif self == ScorecardProfilePath.QNN:
            devices = [
                ScorecardDevices.cs_8_gen_2,
                ScorecardDevices.cs_8_gen_3,
                ScorecardDevices.cs_x_elite,
                ScorecardDevices.cs_8550,
                ScorecardDevices.cs_auto_lemans_8650,
                ScorecardDevices.cs_auto_lemans_8775,
                ScorecardDevices.cs_auto_lemans_8255,
                ScorecardDevices.cs_auto_makena_8295,
                ScorecardDevices.cs_xr_8450,
            ] + ([ScorecardDevices.cs_6490] if aimet_model else [])
        elif self == ScorecardProfilePath.ONNX_DML_GPU:
            devices = [ScorecardDevices.cs_x_elite]
        else:
            raise NotImplementedError()

        try:
            from qai_hub_models.utils.scorecard._common_private import (
                get_private_profile_path_test_devices,
            )

            devices.extend(get_private_profile_path_test_devices(self, aimet_model))  # type: ignore
        except ImportError:
            pass

        return [x for x in devices if x.enabled] if only_enabled else devices

    def get_job_cache_name(
        self,
        model: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ):
        return get_job_cache_name(self.name, model, device, component)


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

    if "qualcomm-snapdragon-x-elite" in chipset_set:
        chipset_list.extend(["qualcomm-snapdragon-x-plus-8-core"])

    chipset_order = [
        "qualcomm-snapdragon-x-elite",
        "qualcomm-qcs6490",
        "qualcomm-qcs8250",
        "qualcomm-qcs8550",
        "qualcomm-sa8775p",
        "qualcomm-sa8650p",
        "qualcomm-sa8255p",
        "qualcomm-qcs8450",
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


try:
    # Register private devices
    # This must live at the end of this file to avoid circular import problems.
    from qai_hub_models.utils.scorecard._common_private import (  # noqa: F401
        PrivateScorecardDevices,
    )
except ImportError:
    pass
