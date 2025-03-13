# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.facemap_3dmm.demo import facemap_3dmm_demo
from qai_hub_models.models.facemap_3dmm_quantized.model import (
    MODEL_ID,
    FaceMap_3DMMQuantizable,
)


def main(is_test: bool = False):
    facemap_3dmm_demo(FaceMap_3DMMQuantizable, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
