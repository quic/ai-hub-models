# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.imagenet_classifier.demo import imagenet_demo
from qai_hub_models.models.convnext_tiny_w8a16_quantized.model import (
    MODEL_ID,
    ConvNextTinyW8A16Quantizable,
)


def main(is_test: bool = False):
    imagenet_demo(ConvNextTinyW8A16Quantizable, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
