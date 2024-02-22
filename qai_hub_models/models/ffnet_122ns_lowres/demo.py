# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
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
