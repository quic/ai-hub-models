# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from enum import unique
from functools import cached_property
from typing import Optional

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.utils.base_config import ParseableQAIHMEnum

_DEVICE_CACHE: dict[str, hub.Device] = {}
UNIVERSAL_DEVICE_SCORECARD_NAME = "universal"


def _get_cached_device(device_name: str) -> hub.Device:
    # Gets a device with attributes & OS. This only comes from hub.get_devices()
    device = _DEVICE_CACHE.get(device_name, None)
    if not device:
        device = hub.get_devices(device_name)[0]
        _DEVICE_CACHE[device_name] = device
    return device


class ScorecardDevice:
    _registry: dict[str, "ScorecardDevice"] = {}

    @classmethod
    def get(cls, device_name: str):
        return [
            x
            for x in ScorecardDevice.all_devices()
            if x.reference_device_name == device_name
            or x.execution_device_name == device_name
        ][0]

    @classmethod
    def all_devices(
        cls,
        enabled: Optional[bool] = None,
        npu_supports_precision: Optional[Precision] = None,
        supports_compile_path: Optional[ScorecardCompilePath] = None,
        supports_profile_path: Optional[ScorecardProfilePath] = None,
        form_factors: Optional[list["ScorecardDevice.FormFactor"]] = None,
        is_mirror: Optional[bool] = None,
    ):
        """
        Get all devices that match the given attributes.
        If an attribute is None, it is ignored when filtering devices.
        """
        return [
            device
            for device in cls._registry.values()
            if (
                (enabled is None or enabled == device.enabled)
                and (
                    npu_supports_precision is None
                    or device.npu_supports_precision(npu_supports_precision)
                )
                and (
                    supports_compile_path is None
                    or supports_compile_path in device.compile_paths
                )
                and (
                    supports_profile_path is None
                    or supports_profile_path in device.profile_paths
                )
                and (form_factors is None or device.form_factor in form_factors)
                and (is_mirror is None or bool(device.mirror_device) == is_mirror)
            )
        ]

    @unique
    class FormFactor(ParseableQAIHMEnum):
        PHONE = 0  # YAML string: Phone
        TABLET = 1  # YAML string: Tablet
        AUTO = 2  # YAML string: Auto
        XR = 3  # YAML string: XR
        COMPUTE = 4  # YAML string: Compute
        IOT = 5  # YAML string: IoT

        @staticmethod
        def from_string(string: str) -> "ScorecardDevice.FormFactor":
            ff = ScorecardDevice.FormFactor[string.upper()]
            if ff == ScorecardDevice.FormFactor.XR:
                assert (
                    string == "XR"
                ), f"Form Factor for XR should be 'XR'. Got {string} instead."
            elif ff == ScorecardDevice.FormFactor.IOT:
                assert (
                    string == "IoT"
                ), f"Form Factor for IoT should be 'IoT'. Got {string} instead."
            else:
                assert (
                    string == string.title()
                ), f"Form Factor should be Title Case ({string.title()}, but got {string} instead."
            return ScorecardDevice.FormFactor[string.upper()]

        def __str__(self):
            if self == ScorecardDevice.FormFactor.IOT:
                return "IoT"
            elif self == ScorecardDevice.FormFactor.XR:
                return "XR"
            return self.name.title()

    @unique
    class OperatingSystem(ParseableQAIHMEnum):
        ANDROID = 0
        WINDOWS = 1
        LINUX = 2

        @staticmethod
        def from_string(string: str) -> "ScorecardDevice.OperatingSystem":
            return ScorecardDevice.OperatingSystem[string.upper()]

        def __str__(self):
            return self.name

    def __init__(
        self,
        name: str,
        reference_device_name: Optional[str],
        execution_device_name: Optional[str] = None,
        disabled_models: list[str] = [],
        compile_paths: Optional[list[ScorecardCompilePath]] = None,
        profile_paths: Optional[list[ScorecardProfilePath]] = None,
        mirror_device: "Optional[ScorecardDevice]" = None,
        public: bool = True,
    ):
        """
        Parameters
            name: Name of this device for scorecard use.

            reference_device_name: The name of the "reference" device used by the scorecard for metadata when collating results.

            execution_device_name: The name of the device to be used by associated Hub jobs.
                                   If not provided, jobs will be submitted with the chipset of the reference device.
                                   Hub will decide what device to use depending on availability.

            compile_paths: The set of compile paths valid for this device. If unset, will use the default set of paths for this device's form factor.

            profile_paths: The set of profile paths valid for this device. If unset, will use the default set of paths for this device's form factor.

            mirror_device: If set, jobs are not run on this device. Instead, results for this will "mirror" of the given device.

            public: Whether this device is publicly available.
        """
        if name in ScorecardDevice._registry:
            raise ValueError("Device " + name + "already registered.")

        if mirror_device:
            assert (
                not compile_paths
            ), "Compile paths should not be set, mirror devices will use the mirror device settings."
            assert (
                not profile_paths
            ), "Profile paths should not be set, mirror devices will use the mirror device settings."
            assert (
                not disabled_models
            ), "Disabled models should not be set, mirror devices will use the mirror device settings."
            assert (
                not execution_device_name
            ), "Execution device is not applicable when mirroring results of a different device."

        self.name = name
        self.disabled_models: list[str] = (
            mirror_device.disabled_models if mirror_device else disabled_models
        )
        self.reference_device_name = reference_device_name
        self.execution_device_name = execution_device_name
        self._compile_paths = compile_paths
        self._profile_paths = profile_paths
        self.mirror_device: Optional[ScorecardDevice] = mirror_device
        self.public = public

        ScorecardDevice._registry[name] = self

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
        valid_test_devices = os.environ.get("QAIHM_TEST_DEVICES", "ALL")
        return (
            valid_test_devices == "ALL"
            or self.name == UNIVERSAL_DEVICE_SCORECARD_NAME
            or self.name in valid_test_devices.split(",")
        )

    @cached_property
    def reference_device(self) -> hub.Device:
        """
        Get the "reference" device used by the scorecard for metadata when collating results.
        This is not used by any actual scorecard jobs.
        """
        if self.reference_device_name is not None:
            return _get_cached_device(self.reference_device_name)
        raise NotImplementedError(f"No reference device for {self.name}")

    @cached_property
    def execution_device(self) -> hub.Device:
        """
        Get the "reference" device used by the scorecard for metadata when collating results.
        This is not used by any actual scorecard jobs.
        """
        if self.execution_device_name is not None:
            return _get_cached_device(self.execution_device_name)
        return self.reference_device

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
    def os(self) -> OperatingSystem:
        """
        The operating system used by this device.
        """
        for attr in self.reference_device.attributes:
            if attr.startswith("os:"):
                return ScorecardDevice.OperatingSystem.from_string(attr[3:])
        raise ValueError(f"OS not found for device: {self.name}")

    @cached_property
    def form_factor(self) -> FormFactor:
        """
        The chipset form_factor (eg. Auto, IoT, Mobile, ...)
        """
        for attr in self.reference_device.attributes:
            if attr.startswith("format:"):
                return ScorecardDevice.FormFactor[attr[7:].upper()]
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

    @cached_property
    def supports_fp16_npu(self) -> bool:
        """
        Whether this device's NPU supports FP16 inference.
        """
        if self.mirror_device:
            return self.mirror_device.supports_fp16_npu

        return "htp-supports-fp16:true" in self.reference_device.attributes

    def npu_supports_precision(self, precision: Precision) -> bool:
        """
        Whether this device's NPU supports the given quantization spec.
        """
        if self.mirror_device:
            return self.mirror_device.npu_supports_precision(precision)

        return not precision.has_float_activations or self.supports_fp16_npu

    @cached_property
    def supported_runtimes(self) -> list[TargetRuntime]:
        if self.mirror_device:
            return self.mirror_device.supported_runtimes

        runtimes = []
        for attr in self.reference_device.attributes:
            if attr.startswith("framework:"):
                rt_name = attr[10:].upper()
                try:
                    runtimes.append(TargetRuntime[rt_name])
                except KeyError:
                    print(
                        f"WARNING: Unable to determine supported runtime associated with framework {rt_name}"
                    )
        return runtimes

    @cached_property
    def profile_paths(self) -> list[ScorecardProfilePath]:
        if self.mirror_device:
            return self.mirror_device.profile_paths

        if self._profile_paths is not None:
            return self._profile_paths

        paths: list[ScorecardProfilePath]
        if self.form_factor in [
            ScorecardDevice.FormFactor.PHONE,
            ScorecardDevice.FormFactor.TABLET,
        ]:
            paths = [
                ScorecardProfilePath.ONNX,
                ScorecardProfilePath.QNN,
                ScorecardProfilePath.TFLITE,
            ]
        elif self.form_factor == ScorecardDevice.FormFactor.AUTO:
            paths = [ScorecardProfilePath.QNN, ScorecardProfilePath.TFLITE]
        elif self.form_factor == ScorecardDevice.FormFactor.XR:
            paths = [ScorecardProfilePath.QNN, ScorecardProfilePath.TFLITE]
        elif self.form_factor == ScorecardDevice.FormFactor.COMPUTE:
            paths = [
                ScorecardProfilePath.ONNX,
                ScorecardProfilePath.ONNX_DML_GPU,
                ScorecardProfilePath.QNN,
            ]
        elif self.form_factor == ScorecardDevice.FormFactor.IOT:
            paths = [ScorecardProfilePath.TFLITE, ScorecardProfilePath.QNN]
        else:
            raise NotImplementedError(
                f"Unsupported device form factor: {self.form_factor}"
            )

        return [path for path in paths if path.runtime in self.supported_runtimes]

    @cached_property
    def compile_paths(self) -> list[ScorecardCompilePath]:
        if self.mirror_device:
            return self.mirror_device.compile_paths

        if self._compile_paths is not None:
            return self._compile_paths

        if ScorecardProfilePath.QNN in self.profile_paths:
            paths = [ScorecardCompilePath.QNN]
        else:
            paths = []

        return [path for path in paths if path.runtime in self.supported_runtimes]


# ----------------------
# DEVICE DEFINITIONS
#
# This list is the set of devices we use by default when benchmarking models.
#
# Typically we define one device per chipset, and devices are named after that chipset:
# cs_8_gen_3 == device representative of the 8 gen 3 chipset
# ----------------------

##
# Universal Chipset
#
# A placeholder for compiling universal assets (that are applicable to any device)
#
# .tflite and .onnx are always universal, so they are compiled once for this device
# and used for inference on all other devices.
#
# This device also produces an android-arm64 QNN .so that is used for inference on all Android devices.
##
cs_universal = ScorecardDevice(
    name=UNIVERSAL_DEVICE_SCORECARD_NAME,
    reference_device_name="Samsung Galaxy S23",
    compile_paths=[
        ScorecardCompilePath.TFLITE,
        ScorecardCompilePath.ONNX,
        ScorecardCompilePath.ONNX_FP16,
    ],
    profile_paths=[],
)


##
# Mobile Chipsets (cs)
##
cs_8_gen_2 = ScorecardDevice(
    name="cs_8_gen_2",
    reference_device_name="Samsung Galaxy S23",
    execution_device_name="Samsung Galaxy S23 (Family)",
)

cs_8_gen_3 = ScorecardDevice(
    name="cs_8_gen_3",
    reference_device_name="Samsung Galaxy S24",
    execution_device_name="Samsung Galaxy S24 (Family)",
)

cs_8_elite = ScorecardDevice(
    name="cs_8_elite", reference_device_name="Snapdragon 8 Elite QRD"
)


##
# Compute Chipsets (cs)
##
cs_x_elite = ScorecardDevice(
    name="cs_x_elite", reference_device_name="Snapdragon X Elite CRD"
)


##
# Auto Chipsets (cs)
##
cs_auto_monaco_7255 = ScorecardDevice(
    name="cs_auto_monaco_7255",
    reference_device_name="SA7255P ADP",
)

cs_auto_lemans_8255 = ScorecardDevice(
    name="cs_auto_lemans_8255",
    reference_device_name="SA8255 (Proxy)",
)

cs_auto_makena_8295 = ScorecardDevice(
    name="cs_auto_makena_8295",
    reference_device_name="SA8295P ADP",
)

cs_auto_lemans_8650 = ScorecardDevice(
    name="cs_auto_lemans_8650",
    reference_device_name="SA8650 (Proxy)",
)

cs_auto_lemans_8775 = ScorecardDevice(
    name="cs_auto_lemans_8775",
    reference_device_name="SA8775P ADP",
)


##
# IoT Chipsets (cs)
##
cs_6490 = ScorecardDevice(
    name="cs_6490",
    reference_device_name="RB3 Gen 2 (Proxy)",
)

cs_8250 = ScorecardDevice(
    name="cs_8250",
    reference_device_name="RB5 (Proxy)",
)

cs_8275 = ScorecardDevice(
    name="cs_8275",
    reference_device_name="QCS8275 (Proxy)",
    mirror_device=cs_auto_monaco_7255,
)

cs_8550 = ScorecardDevice(name="cs_8550", reference_device_name="QCS8550 (Proxy)")


cs_9075 = ScorecardDevice(
    name="cs_9075",
    reference_device_name="QCS9075 (Proxy)",
    mirror_device=cs_auto_lemans_8775,
)


##
# XR Chipsets (cs)
##
cs_xr_8450 = ScorecardDevice(name="cs_xr_8450", reference_device_name="QCS8450 (Proxy)")
