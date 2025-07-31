# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.cityscapes_segmentation.demo import (
    cityscapes_segmentation_demo,
)
from qai_hub_models.models.hrnet_w48_ocr.model import HRNET_W48_OCR, MODEL_ID


def main(is_test: bool = False):
    cityscapes_segmentation_demo(HRNET_W48_OCR, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
