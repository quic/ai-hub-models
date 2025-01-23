# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.repaint.utils import preprocess_inputs
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    "aotgan", 2, "test_images/test_input_image.png"
)
MASK_ADDRESS = CachedWebModelAsset.from_asset_store(
    "aotgan", 2, "test_images/test_input_mask.png"
)


class RepaintModel(BaseModel):
    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "image": ((batch_size, 3, height, width), "float32"),
            "mask": ((batch_size, 1, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["painted_image"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image", "mask"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["painted_image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        """
        Provides an example image of a man with a mask over the glasses.
        """
        image = load_image(IMAGE_ADDRESS)
        mask = load_image(MASK_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
            mask = mask.resize((w, h))
        torch_inputs = preprocess_inputs(image, mask)
        return {k: [v.detach().numpy()] for k, v in torch_inputs.items()}
