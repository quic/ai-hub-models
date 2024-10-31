# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.foot_track_net.demo import demo
from qai_hub_models.models.foot_track_net.model import FootTrackNet


def main(is_test: bool = False):
    demo(FootTrackNet, is_test)


if __name__ == "__main__":
    main()
