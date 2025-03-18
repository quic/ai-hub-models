# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

import numpy as np
import qai_hub as hub
from qai_hub.client import QuantizeDtype
from typing_extensions import assert_never


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
    def channel_last_native_execution(self) -> bool:
        """
        If true, this runtime natively executes ops in NHWC (channel-last) format.
        If false, the runtime executes ops in NCHW (channel-first) format.
        """
        return self in [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ]

    def supports_precision(self, precision: Precision) -> bool:
        # Float is always supported.
        if precision == Precision.float:
            return True

        # Check quantized types
        if self == TargetRuntime.TFLITE:
            return precision == Precision.w8a8
        if self == TargetRuntime.ONNX:
            return precision in [
                Precision.w8a8,
                Precision.w8a16,
                # The following three are enabled tentatively
                # (not experimentally verified)
                Precision.w16a16,
                Precision.w4a16,
                Precision.w4,
            ]
        if self == TargetRuntime.QNN or self == TargetRuntime.PRECOMPILED_QNN_ONNX:
            return precision in [
                Precision.w8a8,
                Precision.w8a16,
                Precision.w4a16,
                Precision.w4,
                Precision.w16a16,
            ]

        assert_never(self)

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


class Precision:
    """
    Stores a precision configuration for a model's graph.

    Precision can represent any precision that AI Hub's QuantizeJob supports.
    It also can represent floating point (no quantization).

    Mixed precision is not currently representable by this class.
    """

    # Common Precision Instances
    float: Precision
    w8a8: Precision
    w8a16: Precision
    w16a16: Precision
    w4a16: Precision
    w4: Precision

    def __init__(
        self, weights_type: QuantizeDtype | None, activations_type: QuantizeDtype | None
    ):
        self.weights_type: QuantizeDtype | None = weights_type
        self.activations_type: QuantizeDtype | None = activations_type

    @staticmethod
    def from_string(string: str) -> Precision:
        if string == "float":
            return Precision.float

        atype = None
        wtype = None
        if match := re.match(r"(w\d+)?(a\d+)?", string):
            wtype = match.group(1)
            atype = match.group(2)
        if match := re.match(r"(a\d+)?(w\d+)?", string):
            atype = atype or match.group(1)
            wtype = wtype or match.group(2)

        if not atype and not wtype:
            raise ValueError(
                "Unsupported quantization type string. Example valid strings: float, w8a16, a8w8, w8, a16, ..."
            )

        atype_enum_name = f"INT{atype[1:]}" if atype else None
        wtype_enum_name = f"INT{wtype[1:]}" if wtype else None
        for enum_name, bit_width, name in [
            (atype_enum_name, atype, "activations"),
            (wtype_enum_name, wtype, "weights"),
        ]:
            if not bit_width:
                continue
            if enum_name not in QuantizeDtype._member_names_:
                raise ValueError(
                    f"Unsupported bit width {bit_width} for quantization {name}. Supported: {','. join([str(x) for x in QuantizeDtype])}"
                )

        return Precision(
            QuantizeDtype[wtype_enum_name] if wtype_enum_name else None,
            QuantizeDtype[atype_enum_name] if atype_enum_name else None,
        )

    @property
    def has_float_activations(self) -> bool:
        # Returns true if model activations are not quantized.
        return self.activations_type is None

    @property
    def has_float_weights(self) -> bool:
        # Returns true if model weights are not quantized.
        return self.weights_type is None

    def __str__(self) -> str:
        if self == Precision.float:
            return "float"

        return (f"w{self.weights_type.name[3:]}" if self.weights_type else "") + (
            f"a{self.activations_type.name[3:]}" if self.activations_type else ""
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        if not isinstance(value, Precision):
            return False
        return (
            value.activations_type == self.activations_type
            and value.weights_type == self.weights_type
        )

    def __hash__(self) -> int:
        return hash(str(self))


Precision.float = Precision(None, None)
Precision.w8a8 = Precision(QuantizeDtype.INT8, QuantizeDtype.INT8)
Precision.w8a16 = Precision(QuantizeDtype.INT8, QuantizeDtype.INT16)
Precision.w16a16 = Precision(QuantizeDtype.INT16, QuantizeDtype.INT16)
Precision.w4a16 = Precision(QuantizeDtype.INT4, QuantizeDtype.INT16)
Precision.w4 = Precision(QuantizeDtype.INT4, None)


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
