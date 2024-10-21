# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import re
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
    if isinstance(val, ScorecardDevice):
        return val.name


class ScorecardDevice:
    # -- DEVICE REGISTRY --
    _registry: Dict[str, "ScorecardDevice"] = {}

    @classmethod
    def all_devices(cls, only_enabled: bool = False) -> List["ScorecardDevice"]:
        if only_enabled:
            return cls.all_enabled()
        return list(cls._registry.values())

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
        duplicate_of: Optional["ScorecardDevice"] = None,
        compile_paths: Optional[List["ScorecardCompilePath"]] = None,
        profile_paths: Optional[List["ScorecardProfilePath"]] = None,
    ) -> "ScorecardDevice":
        if name in cls._registry:
            raise ValueError("Device " + name + "already registered.")

        device = ScorecardDevice(
            name,
            reference_device_name,
            execution_device_name,
            disabled_models,
            duplicate_of,
            compile_paths,
            profile_paths,
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
        duplicate_of: Optional["ScorecardDevice"] = None,
        compile_paths: Optional[List["ScorecardCompilePath"]] = None,
        profile_paths: Optional[List["ScorecardProfilePath"]] = None,
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

            duplicate_of: If set, this device will act as a duplicate of the given scorecard device. In effect this means:
                          * Jobs will not be submitted targeting this chipset.
                          * Jobs for the "given" scorecard device will be used to create performance metrics for this device.

                          NOTE: Just because this chip is marked as having duplicate AI/ML performance compared to another chip,
                                does not mean this chip is indistinguishable from that other chip. The chips will
                                differ by other important features, but these are not relevant for this AI/ML scorecard.

            compile_paths: The set of compile paths valid for this device. If unset, will use the default set of paths for this device type.

            profile_paths: The set of profile paths valid for this device. If unset, will use the default set of paths for this device type.

        """
        self.name = name
        self.disabled_models = disabled_models
        self.reference_device_name = reference_device_name
        self.execution_device_name = execution_device_name
        self.duplicate_of = duplicate_of
        self._compile_paths = compile_paths
        self._profile_paths = profile_paths

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
            or self.name == "any"
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
        raise ValueError(f"OS not found for device: {self.name}")

    @cached_property
    def form_factor(self) -> str:
        """
        The chipset form_factor (eg. Auto, IoT, Mobile, ...)
        """
        for attr in self.reference_device.attributes:
            if attr.startswith("format:"):
                return attr[7:]
        raise ValueError(f"Format not found for device: {self.name}")

    @cached_property
    def hexagon_version(self) -> int:
        """
        The chipset hexagon version number
        """
        for attr in self.reference_device.attributes:
            if attr.startswith("hexagon:v"):
                return int(attr[9:])
        raise ValueError(f"Hexagon version not found for device: {self.name}")

    @property
    def supports_fp16_inference(self) -> bool:
        return self.hexagon_version >= 69

    @cached_property
    def supported_runtimes(self) -> List[TargetRuntime]:
        runtimes = []
        for attr in self.reference_device.attributes:
            if attr.startswith("framework:"):
                rt_name = attr[10:].upper()
                try:
                    runtimes.append(TargetRuntime[rt_name.upper()])
                except KeyError:
                    print(
                        f"WARNING: Unable to determine supported runtime associated with framework {rt_name}"
                    )
        return runtimes

    @cached_property
    def profile_paths(self) -> List["ScorecardProfilePath"]:
        if self._profile_paths:
            return self._profile_paths
        if self.duplicate_of:
            return self.duplicate_of.profile_paths

        if self.form_factor == "phone":
            paths = [
                ScorecardProfilePath.ONNX,
                ScorecardProfilePath.QNN,
                ScorecardProfilePath.TFLITE,
            ]
        elif self.form_factor == "auto":
            paths = [
                ScorecardProfilePath.QNN,
                ScorecardProfilePath.TFLITE,
            ]
        elif self.form_factor == "xr":
            paths = [ScorecardProfilePath.QNN, ScorecardProfilePath.TFLITE]
        elif self.form_factor == "compute":
            paths = [
                ScorecardProfilePath.ONNX,
                ScorecardProfilePath.ONNX_DML_GPU,
                ScorecardProfilePath.QNN,
            ]
        elif self.form_factor == "iot":
            paths = [ScorecardProfilePath.TFLITE, ScorecardProfilePath.QNN]
        else:
            raise NotImplementedError(
                f"Unsupported device form_factor: {self.form_factor}"
            )

        return [path for path in paths if path.get_runtime() in self.supported_runtimes]

    @cached_property
    def compile_paths(self) -> List["ScorecardCompilePath"]:
        if self._compile_paths:
            return self._compile_paths
        if self.duplicate_of:
            return self.duplicate_of.compile_paths

        if ScorecardProfilePath.QNN in self.profile_paths:
            paths = [ScorecardCompilePath.QNN]
        else:
            paths = []

        return [path for path in paths if path.get_runtime() in self.supported_runtimes]


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
        is_quantized=False,
        only_enabled_paths=True,
        only_enabled_devices=True,
    ) -> List[Tuple["ScorecardCompilePath", ScorecardDevice]]:
        path_list: List[ScorecardCompilePath] = ScorecardCompilePath.all_enabled() if only_enabled_paths else ScorecardCompilePath  # type: ignore
        path_devices_dict = {
            sc_path: sc_path.get_test_devices(
                is_quantized, only_enabled_devices, include_any=True
            )
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
        self,
        is_quantized: bool = False,
        only_enabled: bool = True,
        include_duplicate_devices: bool = False,
        include_any: bool = False,
    ) -> List[ScorecardDevice]:
        return [
            device
            for device in ScorecardDevice.all_devices(only_enabled)
            if (
                (is_quantized or device.supports_fp16_inference)
                and (include_duplicate_devices or not device.duplicate_of)
                and (include_any or device != ScorecardDevices.any)
                and self in device.compile_paths
            )
        ]

    def get_compile_options(self, is_quantized=False) -> str:
        if self == ScorecardCompilePath.ONNX_FP16 and not is_quantized:
            return "--quantize_full_type float16 --quantize_io"
        return ""

    def get_job_cache_name(
        self,
        model: str,
        device: Optional[ScorecardDevice] = None,
        is_quantized: bool = False,
        component: Optional[str] = None,
    ):
        if not device or self not in device.compile_paths:
            device = ScorecardDevices.any  # default to the "generic" compilation path
        return get_job_cache_name(
            self.name, model, device.duplicate_of or device, component
        )


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
        is_quantized=False,
        only_enabled_paths=True,
        only_enabled_devices=True,
    ) -> List[Tuple["ScorecardProfilePath", ScorecardDevice]]:
        path_list: List[ScorecardProfilePath] = ScorecardProfilePath.all_enabled() if only_enabled_paths else ScorecardProfilePath  # type: ignore
        path_devices_dict = {
            sc_path: sc_path.get_test_devices(
                is_quantized, only_enabled_devices, include_any=True
            )
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
        self,
        is_quantized: bool = False,
        only_enabled: bool = True,
        include_duplicate_devices: bool = False,
        include_any: bool = False,
    ) -> List[ScorecardDevice]:
        return [
            device
            for device in ScorecardDevice.all_devices(only_enabled)
            if (
                (is_quantized or device.supports_fp16_inference)
                and (include_duplicate_devices or not device.duplicate_of)
                and self in device.profile_paths
                and (include_any or device != ScorecardDevices.any)
            )
        ]

    def get_job_cache_name(
        self,
        model: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ):
        return get_job_cache_name(
            self.name, model, device.duplicate_of or device, component
        )


def supported_chipsets(chips: List[str]) -> List[str]:
    """
    Return all the supported chipsets given the chipset it works on.

    The order of chips in the website list mirror the order here. Order
    chips from newest to oldest to highlight newest chips most prominently.
    """
    chipset_set = set(chips)
    chipset_list = []

    if "qualcomm-snapdragon-8-elite" in chipset_set:
        chipset_list.extend(
            [
                "qualcomm-snapdragon-8-elite",
                "qualcomm-snapdragon-8gen3",
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
        )
    elif "qualcomm-snapdragon-8gen3" in chipset_set:
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
        chipset_list.extend(["qualcomm-snapdragon-x-elite"])
        chipset_list.extend(["qualcomm-snapdragon-x-plus-8-core"])

    chipset_order = [
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
    for chipset in sorted(chipset_set):
        if chipset not in chipset_list:
            chipset_list.append(chipset)
    return chipset_list


def chipset_marketing_name(chipset) -> str:
    """Sanitize chip name to match marketing."""
    chip = " ".join([word.capitalize() for word in chipset.split("-")])
    chip = chip.replace("Qualcomm ", "")
    chip = chip.replace(
        "Snapdragon", "Snapdragon®"
    )  # Marketing name for Qualcomm Snapdragon is Snapdragon®

    # 8cxgen2 -> 8cx Gen 2
    # 8gen2 -> 8 Gen 2
    chip = re.sub(r"(\w+)gen(\d+)", r"\g<1> Gen \g<2>", chip)

    # 8 Core -> 8-Core
    chip = re.sub(r"(\d+) Core", r"\g<1>-Core", chip)

    # qcs6490 -> QCS6490
    # sa8775p -> SA8775P
    chip = re.sub(
        r"(Qcs|Sa)(\w+)", lambda m: f"{m.group(1).upper()}{m.group(2).upper()}", chip
    )

    return chip


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
    return supported_devices


def supported_oses() -> List[str]:
    """Return all the supported operating systems."""
    return ["Android"]


class ScorecardDevices:
    any = ScorecardDevice.register(
        name="any",
        reference_device_name="Samsung Galaxy S23",
        compile_paths=[path for path in ScorecardCompilePath],
        profile_paths=[],
    )  # no specific device (usable only during compilation)

    ###
    # cs == chipset
    ###
    cs_8_gen_2 = ScorecardDevice.register(
        name="cs_8_gen_2",
        reference_device_name="Samsung Galaxy S23",
        compile_paths=[],  # Uses "any" in all cases
    )

    cs_8_gen_3 = ScorecardDevice.register(
        name="cs_8_gen_3",
        reference_device_name="Samsung Galaxy S24",
        execution_device_name="Samsung Galaxy S24 (Family)",
        compile_paths=[],  # Uses "any" in all cases
    )

    cs_6490 = ScorecardDevice.register(
        name="cs_6490",
        reference_device_name="RB3 Gen 2 (Proxy)",
        disabled_models=[
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

    cs_8250 = ScorecardDevice.register(
        name="cs_8250",
        reference_device_name="RB5 (Proxy)",
    )

    cs_8550 = ScorecardDevice.register(
        name="cs_8550", reference_device_name="QCS8550 (Proxy)"
    )

    cs_x_elite = ScorecardDevice.register(
        name="cs_x_elite", reference_device_name="Snapdragon X Elite CRD"
    )

    cs_auto_lemans_8255 = ScorecardDevice.register(
        name="cs_auto_lemans_8255",
        reference_device_name="SA8255 (Proxy)",
    )

    cs_auto_lemans_8775 = ScorecardDevice.register(
        name="cs_auto_lemans_8775",
        reference_device_name="SA8775 (Proxy)",
    )

    cs_auto_lemans_8650 = ScorecardDevice.register(
        name="cs_auto_lemans_8650",
        reference_device_name="SA8650 (Proxy)",
    )

    cs_xr_8450 = ScorecardDevice.register(
        name="cs_xr_8450", reference_device_name="QCS8450 (Proxy)"
    )

    cs_8_elite = ScorecardDevice.register(
        name="cs_8_elite", reference_device_name="Snapdragon 8 Elite QRD"
    )


try:
    # Register private devices
    # This must live at the end of this file to avoid circular import problems.
    from qai_hub_models.utils.scorecard._common_private import (  # noqa: F401
        PrivateScorecardDevices,
    )
except ImportError:
    pass
