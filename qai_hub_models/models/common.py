# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from enum import Enum
from typing import Dict, List

import numpy as np


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


class SourceModelFormat(Enum):
    ONNX = 0
    TORCHSCRIPT = 1


SampleInputsType = Dict[str, List[np.ndarray]]
