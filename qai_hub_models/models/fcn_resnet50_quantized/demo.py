# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.fcn_resnet50.demo import fcn_resnet50_demo
from qai_hub_models.models.fcn_resnet50_quantized.model import FCN_ResNet50Quantizable


def main(is_test: bool = False):
    fcn_resnet50_demo(FCN_ResNet50Quantizable, is_test)


if __name__ == "__main__":
    main()
