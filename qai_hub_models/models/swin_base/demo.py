# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.imagenet_classifier.demo import imagenet_demo
from qai_hub_models.models.swin_base.model import MODEL_ID, SwinBase


def main(is_test: bool = False):
    imagenet_demo(SwinBase, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
