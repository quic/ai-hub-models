# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.midas.demo import midas_demo
from qai_hub_models.models.midas_quantized.model import MidasQuantizable


def main(is_test: bool = False):
    midas_demo(MidasQuantizable, is_test)


if __name__ == "__main__":
    main()
