# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.repaint.demo import repaint_demo
from qai_hub_models.models.aotgan.model import (
    AOTGAN,
    IMAGE_ADDRESS,
    MASK_ADDRESS,
    MODEL_ID,
)


def main(is_test: bool = False):
    repaint_demo(AOTGAN, MODEL_ID, IMAGE_ADDRESS, MASK_ADDRESS, is_test=is_test)


if __name__ == "__main__":
    main()
