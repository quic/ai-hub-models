# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.video_classifier.demo import kinetics_classifier_demo
from qai_hub_models.models._shared.video_classifier.model import INPUT_VIDEO_PATH
from qai_hub_models.models.resnet_mixed_quantized.model import ResNetMixedQuantizable


def main(is_test: bool = False):
    kinetics_classifier_demo(
        model_type=ResNetMixedQuantizable,
        default_video=INPUT_VIDEO_PATH,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
