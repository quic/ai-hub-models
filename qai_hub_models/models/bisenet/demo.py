# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.segmentation.demo import segmentation_demo
from qai_hub_models.models.bisenet.model import INPUT_IMAGE_ADDRESS, MODEL_ID, BiseNet


def main(is_test: bool = False):
    segmentation_demo(BiseNet, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test, False)


if __name__ == "__main__":
    main()
