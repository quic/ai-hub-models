# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from ultralytics.models import FastSAM
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.ultralytics.segmentation_model import (
    UltralyticsSingleClassSegmentor,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec


class Fast_SAM(UltralyticsSingleClassSegmentor):
    """Exportable FastSAM model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = ""):
        return cls(cast(SegmentationModel, FastSAM(model=ckpt_name).model))

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "fastsam_s", 1, "image_640.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
