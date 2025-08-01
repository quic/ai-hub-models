# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.llm.quantize import llm_quantize
from qai_hub_models.models.llama_v3_2_3b_instruct import MODEL_ID, FP_Model, Model

if __name__ == "__main__":
    llm_quantize(quantized_model_cls=Model, fp_model_cls=FP_Model, model_id=MODEL_ID)
