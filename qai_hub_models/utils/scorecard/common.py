# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from enum import Enum
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


class ScorecardDevice(Enum):
    any = 0  # no specific device (usable only during compilation)

    # cs == chipset
    cs_8_gen_2 = 1
    cs_8_gen_3 = 2
    cs_6490 = 3
    cs_8250 = 4
    cs_8550 = 5
    cs_x_elite = 6
    cs_auto_lemans_8255 = 7
    cs_auto_lemans_8775 = 8
    cs_auto_lemans_8650 = 9
    # cs_auto_makena_8540  | Disabled until fp16 support is enabled for makena.

    def enabled(self) -> bool:
        valid_test_devices = os.environ.get("WHITELISTED_PROFILE_TEST_DEVICES", "ALL")
        return (
            valid_test_devices == "ALL"
            or self == ScorecardDevice.any
            or self.name in valid_test_devices.split(",")
        )

    def get_disabled_models(self) -> List[str]:
        """
        Each chipset can have a list of 'disabled' models, for which the
        chipset won't show up as a 'supported chipset' for that model.
        """
        if self == ScorecardDevice.cs_6490:
            return [
                "ConvNext-Tiny-w8a8-Quantized",
                "ConvNext-Tiny-w8a16-Quantized",
                "ResNet50Quantized",
                "RegNetQuantized",
                "HRNetPoseQuantized",
                "SESR-M5-Quantized",
                "Midas-V2-Quantized",
                "Posenet-Mobilenet-Quantized",
            ]
        return []

    def all_enabled(self) -> List["ScorecardDevice"]:
        return [x for x in ScorecardDevice if x.enabled()]

    def get_reference_device(self) -> hub.Device:
        if self in [ScorecardDevice.cs_8_gen_2, ScorecardDevice.any]:
            return _get_cached_device("Samsung Galaxy S23")
        if self == ScorecardDevice.cs_8_gen_3:
            return _get_cached_device("Samsung Galaxy S24")
        if self == ScorecardDevice.cs_6490:
            return _get_cached_device("RB3 Gen 2 (Proxy)")
        if self == ScorecardDevice.cs_8250:
            return _get_cached_device("RB5 (Proxy)")
        if self == ScorecardDevice.cs_8550:
            return _get_cached_device("QCS8550 (Proxy)")
        if self == ScorecardDevice.cs_x_elite:
            return _get_cached_device("Snapdragon X Elite CRD")
        if self == ScorecardDevice.cs_auto_lemans_8255:
            return _get_cached_device("SA8255 (Proxy)")
        if self == ScorecardDevice.cs_auto_lemans_8775:
            return _get_cached_device("SA8775 (Proxy)")
        if self == ScorecardDevice.cs_auto_lemans_8650:
            return _get_cached_device("SA8650 (Proxy)")
        # if self == ScorecardDevice.cs_auto_makena_8540:
        #    return _get_cached_device("SA8540 (Proxy)")
        raise NotImplementedError(f"No reference device for {self.name}")

    def get_chipset(self) -> str:
        if self in [ScorecardDevice.cs_8_gen_2, ScorecardDevice.any]:
            return "qualcomm-snapdragon-8gen2"
        if self == ScorecardDevice.cs_8_gen_3:
            return "qualcomm-snapdragon-8gen3"
        if self == ScorecardDevice.cs_6490:
            return "qualcomm-qcs6490"
        if self == ScorecardDevice.cs_8250:
            return "qualcomm-qcs8250"
        if self == ScorecardDevice.cs_8550:
            return "qualcomm-qcs8550"
        if self == ScorecardDevice.cs_x_elite:
            return "qualcomm-snapdragon-x-elite"
        if self == ScorecardDevice.cs_auto_lemans_8255:
            return "qualcomm-sa8255p"
        if self == ScorecardDevice.cs_auto_lemans_8775:
            return "qualcomm-sa8775p"
        if self == ScorecardDevice.cs_auto_lemans_8650:
            return "qualcomm-sa8650p"
        # if self == ScorecardDevice.cs_auto_makena_8540:
        #    return "qualcomm-sa8540p"
        raise NotImplementedError(f"No chipset for {self.name}")

    def get_os(self) -> str:
        for attr in self.get_reference_device().attributes:
            if attr.startswith("os:"):
                return attr[3:]
        raise ValueError(f"OS Not found for device: {self.name}")


def get_job_cache_name(
    runtime_name: str,
    model_name: str,
    device: ScorecardDevice,
    component: Optional[str] = None,
) -> str:
    return (
        f"{model_name}_{runtime_name}"
        + ("-" + device.name if device != ScorecardDevice.any else "")
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
                ScorecardDevice.any,
                ScorecardDevice.cs_x_elite,
                ScorecardDevice.cs_8550,
            ]
            if aimet_model:
                devices.append(ScorecardDevice.cs_6490)
        else:
            devices = [ScorecardDevice.any]

        return [x for x in devices if x.enabled()] if only_enabled else devices

    def get_compile_options(self, aimet_model=False) -> str:
        if self == ScorecardCompilePath.ONNX_FP16 and not aimet_model:
            return "--quantize_full_type float16 --quantize_io"
        return ""

    def get_job_cache_name(
        self,
        model: str,
        device: ScorecardDevice = ScorecardDevice.any,
        aimet_model: bool = False,
        component: Optional[str] = None,
    ):
        if device not in self.get_test_devices(aimet_model=aimet_model):
            device = ScorecardDevice.any  # default to the "generic" compilation path
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
                ScorecardDevice.cs_8_gen_2,
                ScorecardDevice.cs_8_gen_3,
                ScorecardDevice.cs_8550,
            ] + (
                [ScorecardDevice.cs_6490, ScorecardDevice.cs_8250]
                if aimet_model
                else []
            )
        elif self == ScorecardProfilePath.ONNX:
            devices = [
                ScorecardDevice.cs_8_gen_2,
                ScorecardDevice.cs_8_gen_3,
                ScorecardDevice.cs_x_elite,
            ]
        elif self == ScorecardProfilePath.QNN:
            devices = [
                ScorecardDevice.cs_8_gen_2,
                ScorecardDevice.cs_8_gen_3,
                ScorecardDevice.cs_x_elite,
                ScorecardDevice.cs_8550,
                ScorecardDevice.cs_auto_lemans_8650,
                ScorecardDevice.cs_auto_lemans_8775,
                ScorecardDevice.cs_auto_lemans_8255,
            ] + ([ScorecardDevice.cs_6490] if aimet_model else [])
        elif self == ScorecardProfilePath.ONNX_DML_GPU:
            devices = [ScorecardDevice.cs_x_elite]
        else:
            raise NotImplementedError()

        return [x for x in devices if x.enabled()] if only_enabled else devices

    def get_job_cache_name(
        self,
        model: str,
        device: ScorecardDevice,
        component: Optional[str] = None,
    ):
        return get_job_cache_name(self.name, model, device, component)
