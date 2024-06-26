# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.posenet_mobilenet.demo import posenet_demo
from qai_hub_models.models.posenet_mobilenet_quantized.model import (
    PosenetMobilenetQuantizable,
)


def main(is_test: bool = False):
    return posenet_demo(PosenetMobilenetQuantizable, is_test)


if __name__ == "__main__":
    main()
