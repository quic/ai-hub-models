# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llama3.quantize import llama3_quantize
from qai_hub_models.models.llama_v3_taide_8b_chat import MODEL_ID, FP_Model, Model

if __name__ == "__main__":
    llama3_quantize(quantized_model_cls=Model, fp_model_cls=FP_Model, model_id=MODEL_ID)
