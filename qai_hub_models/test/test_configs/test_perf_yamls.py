# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.utils.path_helpers import MODEL_IDS


def test_perf_yaml():
    for model_id in MODEL_IDS:
        try:
            QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        except Exception as err:
            assert False, f"{model_id} perf yaml validation failed: {str(err)}"
