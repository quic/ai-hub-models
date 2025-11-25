# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import warnings

from qai_hub_models.models._shared.llm.export import export_model
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.qwen2_5_7b_instruct import MODEL_ID, Model
from qai_hub_models.models.qwen2_5_7b_instruct.model import (
    MODEL_ASSET_VERSION,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
)
from qai_hub_models.utils.args import export_parser

DEFAULT_EXPORT_DEVICE = "Snapdragon 8 Elite QRD"


def main():
    warnings.filterwarnings("ignore")
    parser = export_parser(
        model_cls=Model,
        export_fn=export_model,
        supported_precision_runtimes={Precision.w8a16: [TargetRuntime.GENIE]},
        default_export_device=DEFAULT_EXPORT_DEVICE,
    )
    args = parser.parse_args()
    export_model(
        model_cls=Model,
        model_name=MODEL_ID,
        model_asset_version=MODEL_ASSET_VERSION,
        num_splits=NUM_SPLITS,
        num_layers_per_split=NUM_LAYERS_PER_SPLIT,
        **vars(args),
    )


if __name__ == "__main__":
    main()
