# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.cityscapes_segmentation.demo import (
    cityscapes_segmentation_demo,
)
from qai_hub_models.models.ffnet_122ns_lowres.model import MODEL_ID, FFNet122NSLowRes


def main(is_test: bool = False):
    cityscapes_segmentation_demo(FFNet122NSLowRes, MODEL_ID, is_test=is_test)


if __name__ == "__main__":
    main()
