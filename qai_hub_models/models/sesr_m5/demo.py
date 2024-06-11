# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.super_resolution.demo import super_resolution_demo
from qai_hub_models.models.sesr_m5.model import MODEL_ID, SESR_M5


# Run QuickSRNet end-to-end on a sample image.
# The demo will display an upscaled image
def main(is_test: bool = False):
    super_resolution_demo(
        model_cls=SESR_M5,
        model_id=MODEL_ID,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
