# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Type

import numpy as np
import torch

from qai_hub_models.models._shared.cityscapes_segmentation.demo import (
    TEST_CITYSCAPES_LIKE_IMAGE_ASSET,
)
from qai_hub_models.models._shared.ffnet.model import _load_ffnet_source_model
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import preprocess_PIL_image


def run_test_off_target_numerical(
    ffnet_cls: Type[BaseModel], variant_name: str, relax_numerics: bool = False
):
    """Verify that raw (numeric) outputs of both (qaism and non-qaism) networks are the same."""
    processed_sample_image = preprocess_PIL_image(
        load_image(TEST_CITYSCAPES_LIKE_IMAGE_ASSET)
    )
    source_model = _load_ffnet_source_model(variant_name)
    qaism_model = ffnet_cls.from_pretrained()

    with torch.no_grad():
        source_out = source_model(processed_sample_image)
        qaism_out = qaism_model(processed_sample_image)

        if relax_numerics:
            # At least 90% of pixels should have original prediction
            assert (source_out.argmax(1) == qaism_out.argmax(1)).float().mean() > 0.9
        else:
            np.testing.assert_array_almost_equal(source_out, qaism_out)
