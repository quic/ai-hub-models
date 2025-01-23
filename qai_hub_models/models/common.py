# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

import numpy as np
import qai_hub as hub


@unique
class TargetRuntime(Enum):
    TFLITE = 0
    QNN = 1
    ONNX = 2
    PRECOMPILED_QNN_ONNX = 3

    def __str__(self):
        return self.name.lower()

    @property
    def long_name(self):
        if self.name.lower() in {"tflite", "qnn"}:
            return f"torchscript_onnx_{self.name.lower()}"
        elif self.name.lower() == "onnx":
            return f"torchscript_{self.name.lower()}"
        return f"{self.name.lower()}"

    @property
    def channel_last_native_execution(self):
        """
        If true, this runtime natively executes ops in NHWC (channel-last) format.
        If false, the runtime executes ops in NCHW (channel-first) format.
        """
        return self in [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ]

    def get_target_runtime_flag(
        self,
        device: Optional[hub.Device] = None,
    ) -> str:
        """
        AI Hub job flag for compiling to this runtime.
        """
        target_runtime_flag = None
        if self == TargetRuntime.QNN:
            if device:
                if not device.attributes:
                    # Only name / os specified
                    devices = hub.get_devices(device.name, device.os)
                elif not device.name:
                    # Only attribute specified
                    devices = hub.get_devices(attributes=device.attributes)
                else:
                    devices = [device]

                for device in devices:
                    if (
                        "os:android" not in device.attributes
                        or "format:iot" in device.attributes
                        or "format:auto" in device.attributes
                    ):
                        target_runtime_flag = "qnn_context_binary"
                        break

            target_runtime_flag = target_runtime_flag or "qnn_lib_aarch64_android"
        elif self == TargetRuntime.ONNX:
            target_runtime_flag = "onnx"
        elif self == TargetRuntime.TFLITE:
            target_runtime_flag = "tflite"
        elif self == TargetRuntime.PRECOMPILED_QNN_ONNX:
            target_runtime_flag = "precompiled_qnn_onnx"
        else:
            raise NotImplementedError()

        return f"--target_runtime {target_runtime_flag}"


@unique
class SourceModelFormat(Enum):
    ONNX = 0
    TORCHSCRIPT = 1


SampleInputsType = dict[str, list[np.ndarray]]


@dataclass
class ExportResult:
    compile_job: Optional[hub.CompileJob] = None
    quantize_job: Optional[hub.QuantizeJob] = None
    profile_job: Optional[hub.ProfileJob] = None
    inference_job: Optional[hub.InferenceJob] = None
    link_job: Optional[hub.LinkJob] = None
