# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.super_resolution.demo import super_resolution_demo
from qai_hub_models.models.sesr_m5_quantized.model import MODEL_ID, SESR_M5Quantizable


def main(is_test: bool = False):
    super_resolution_demo(
        SESR_M5Quantizable,
        MODEL_ID,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
