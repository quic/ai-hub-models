# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Collection

import onnxruntime
import torch

from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


def mock_torch_onnx_inference(
    session: onnxruntime.InferenceSession, *args: torch.Tensor, **kwargs: torch.Tensor
) -> torch.Tensor | Collection[torch.Tensor]:
    input_names = [inp.name for inp in session.get_inputs()]
    tensors = tuple(list(args) + list(kwargs.values()))
    input_dict = {
        k: v[0] for k, v in make_hub_dataset_entries(tensors, input_names).items()
    }
    output_np = session.run(None, input_dict)
    output_tensors = [torch.from_numpy(out) for out in output_np]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors
