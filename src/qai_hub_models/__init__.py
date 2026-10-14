# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import sys
import warnings

from .common import (
    ConversionToolchain as ConversionToolchain,
)
from .common import (
    InferenceEngine as InferenceEngine,
)
from .common import (
    Precision as Precision,
)
from .common import (
    QAIRTVersion as QAIRTVersion,
)
from .common import (
    SampleInputsType as SampleInputsType,
)
from .common import (
    TargetRuntime as TargetRuntime,
)
from .common import (
    _FloatDtype as _FloatDtype,
)

if sys.version_info < (3, 11):
    warnings.warn(
        "Python 3.10 support is ending: "
        "qai-hub-models will stop publishing for Python 3.10 starting in January. "
        "Please upgrade to Python 3.12 or newer.",
        FutureWarning,
        stacklevel=2,
    )
