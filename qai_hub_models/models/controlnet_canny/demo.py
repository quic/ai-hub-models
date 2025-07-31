# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

from qai_hub_models.models._shared.stable_diffusion.demo import stable_diffusion_demo
from qai_hub_models.models.controlnet_canny import MODEL_ID, Model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

DEFAULT_PROMPT = "the mona lisa"

DEFAULT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    1,
    os.path.join("demo_asset", "input_image_vermeer.png"),
)

if __name__ == "__main__":
    stable_diffusion_demo(
        MODEL_ID,
        Model,
        use_controlnet=True,
        default_num_steps=20,
        default_prompt=DEFAULT_PROMPT,
        default_image=DEFAULT_IMAGE.fetch(),
    )
