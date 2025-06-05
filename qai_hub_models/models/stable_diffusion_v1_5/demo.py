# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.stable_diffusion.demo import stable_diffusion_demo
from qai_hub_models.models.stable_diffusion_v1_5 import MODEL_ID, Model

if __name__ == "__main__":
    stable_diffusion_demo(MODEL_ID, Model)
