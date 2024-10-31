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
