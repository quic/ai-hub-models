# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.super_resolution.demo import super_resolution_demo
from qai_hub_models.models.quicksrnetlarge_quantized.model import (
    MODEL_ID,
    QuickSRNetLargeQuantizable,
)


def main(is_test: bool = False):
    super_resolution_demo(
        QuickSRNetLargeQuantizable,
        MODEL_ID,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
