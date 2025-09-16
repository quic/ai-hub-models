# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx, LLMBase
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.models.common import Precision


def setup_test_quantization(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    output_path: str,
    precision: Precision,
    checkpoint: str | None = None,
    num_samples: int = 0,
) -> str:
    if not (
        (Path(output_path) / "model.encodings").exists()
        and (Path(output_path) / "model.data").exists()
        and (Path(output_path) / "model_seqlen1_cl4096.onnx").exists()
        and (Path(output_path) / "model_seqlen128_cl4096.onnx").exists()
    ):
        quantize(
            quantized_model_cls=model_cls,
            fp_model_cls=fp_model_cls,
            context_length=4096,
            seq_len=2048,
            precision=precision,
            output_dir=output_path,
            allow_cpu_to_quantize=True,
            checkpoint=checkpoint,
            num_samples=num_samples,
        )

    return output_path
