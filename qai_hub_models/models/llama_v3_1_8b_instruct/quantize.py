# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llama3.quantize import llama3_quantize
from qai_hub_models.models.llama_v3_1_8b_instruct import MODEL_ID, Model
from qai_hub_models.models.llama_v3_1_8b_instruct.model import Llama3_1_8B

if __name__ == "__main__":
    llama3_quantize(
        quantized_model_cls=Model, fp_model_cls=Llama3_1_8B, model_id=MODEL_ID
    )
