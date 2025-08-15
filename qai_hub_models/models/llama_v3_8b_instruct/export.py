# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import warnings

from qai_hub_models.models._shared.llm.export import export_model
from qai_hub_models.models._shared.llm.model import determine_precision_from_checkpoint
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.llama_v3_8b_instruct import (
    MODEL_ID,
    FP_Model,
    Model,
    PositionProcessor,
)
from qai_hub_models.models.llama_v3_8b_instruct.model import (
    DEFAULT_PRECISION,
    MODEL_ASSET_VERSION,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
)
from qai_hub_models.utils.args import enable_model_caching, export_parser

DEFAULT_EXPORT_DEVICE = "Snapdragon 8 Elite QRD"

ALL_COMPONENTS = [f"part_{i + 1}_of_{NUM_SPLITS}" for i in range(NUM_SPLITS)]

# Each components is two sub-components linked together with shared weights
ALL_SUB_COMPONENTS = {
    f"part_{i + 1}_of_{NUM_SPLITS}": [
        f"prompt_{i + 1}_of_{NUM_SPLITS}",
        f"token_{i + 1}_of_{NUM_SPLITS}",
    ]
    for i in range(NUM_SPLITS)
}


def main():
    warnings.filterwarnings("ignore")
    parser = export_parser(
        model_cls=Model,
        supported_precision_runtimes={
            Precision.w4a16: [
                TargetRuntime.QNN_CONTEXT_BINARY,
                TargetRuntime.PRECOMPILED_QNN_ONNX,
            ]
        },
        default_export_device=DEFAULT_EXPORT_DEVICE,
        uses_link_job=True,
    )
    parser.add_argument(
        "--synchronous",
        action="store_true",
        help="Wait for each command to finish before submitting new.",
    )
    parser = enable_model_caching(parser)
    parser.set_defaults(_skip_quantsim_creation=True)
    args = parser.parse_args()
    additional_model_kwargs = vars(args)
    fp_model_params = dict(
        sequence_length=additional_model_kwargs["sequence_length"],
        context_length=additional_model_kwargs["context_length"],
    )
    if isinstance(
        additional_model_kwargs["checkpoint"], str
    ) and additional_model_kwargs["checkpoint"].startswith("DEFAULT"):
        additional_model_kwargs["fp_model"] = FP_Model.from_pretrained(  # type: ignore[index]
            **fp_model_params
        )
        additional_model_kwargs["precision"] = (
            determine_precision_from_checkpoint(additional_model_kwargs["checkpoint"])
            or DEFAULT_PRECISION
        )
    export_model(
        model_cls=Model,
        position_processor_cls=PositionProcessor,
        model_name=MODEL_ID,
        model_asset_version=MODEL_ASSET_VERSION,
        components=ALL_COMPONENTS,
        sub_components=ALL_SUB_COMPONENTS,
        num_layers_per_split=NUM_LAYERS_PER_SPLIT,
        **additional_model_kwargs,
    )


if __name__ == "__main__":
    main()
