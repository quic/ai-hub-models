# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.segmentation.demo import segmentation_demo
from qai_hub_models.models.bisenet.model import INPUT_IMAGE_ADDRESS, MODEL_ID, BiseNet


def main(is_test: bool = False):
    segmentation_demo(BiseNet, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
