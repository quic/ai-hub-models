# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.mediapipe_face.demo import mediapipe_face_demo
from qai_hub_models.models.mediapipe_face_quantized.model import (
    MediaPipeFaceQuantizable,
)


def main(is_test: bool = False):
    return mediapipe_face_demo(MediaPipeFaceQuantizable, is_test)


if __name__ == "__main__":
    main()
