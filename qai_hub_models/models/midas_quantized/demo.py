# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.depth_estimation.demo import depth_estimation_demo
from qai_hub_models.models.midas.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.midas_quantized.model import MODEL_ID, MidasQuantizable


def main(is_test: bool = False):
    depth_estimation_demo(MidasQuantizable, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
