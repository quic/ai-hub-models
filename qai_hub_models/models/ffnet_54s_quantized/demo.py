# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.cityscapes_segmentation.demo import (
    cityscapes_segmentation_demo,
)
from qai_hub_models.models.ffnet_54s_quantized.model import (
    MODEL_ID,
    FFNet54SQuantizable,
)


def main(is_test: bool = False):
    cityscapes_segmentation_demo(FFNet54SQuantizable, MODEL_ID, is_test=is_test)


if __name__ == "__main__":
    main()
