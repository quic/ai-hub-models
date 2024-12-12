# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.face_detection.demo import main as face_det_lite_demo
from qai_hub_models.models.face_det_lite_quantized.model import (
    MODEL_ID,
    FaceDetLiteQuantizable,
)


def main(is_test: bool = False):
    face_det_lite_demo(FaceDetLiteQuantizable, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
