# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.stable_diffusion.quantize import (
    stable_diffusion_quantize,
)
from qai_hub_models.models.stable_diffusion_v2_1 import MODEL_ID, Model

if __name__ == "__main__":
    stable_diffusion_quantize(
        model_cls=Model,
        model_id=MODEL_ID,
        default_num_steps=Model.default_num_steps,
    )
