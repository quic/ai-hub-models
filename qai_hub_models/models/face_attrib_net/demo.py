# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.face_attrib_net.demo import (
    face_attrib_net_demo as demo_main,
)
from qai_hub_models.models.face_attrib_net.model import FaceAttribNet


def main(is_test: bool = False):
    demo_main(FaceAttribNet, is_test)


if __name__ == "__main__":
    main()
