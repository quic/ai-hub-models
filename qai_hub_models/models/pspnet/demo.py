# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models._shared.segmentation.demo import segmentation_demo
from qai_hub_models.models.pspnet.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ID,
    PSPNet,
)


def main(is_test: bool = False):
    segmentation_demo(
        PSPNet, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test, normalize_input=False
    )


if __name__ == "__main__":
    main()
