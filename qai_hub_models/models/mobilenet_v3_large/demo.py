# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.imagenet_classifier.demo import imagenet_demo
from qai_hub_models.models.mobilenet_v3_large.model import MODEL_ID, MobileNetV3Large


def main(is_test: bool = False):
    imagenet_demo(MobileNetV3Large, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
