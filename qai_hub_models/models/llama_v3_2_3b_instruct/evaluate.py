# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llama3.evaluate import llama3_evaluate
from qai_hub_models.models.llama_v3_2_3b_instruct import FP_Model, Model

if __name__ == "__main__":
    llama3_evaluate(quantized_model_cls=Model, fp_model_cls=FP_Model)
