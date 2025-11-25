# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from functools import partial
from typing import cast

import torch
from torch import nn

from qai_hub_models.models._shared.centernet.model_patches import custom_dcn_forward
from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

CenterNetAsRoot = partial(
    SourceAsRoot,
    "https://github.com/xingyizhou/CenterNet.git",
    "4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c",
    "centernet",
    1,
)


class CenterNet(BaseModel):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @classmethod
    def from_pretrained(cls, ckpt_path: str, heads: dict) -> CenterNet:
        with CenterNetAsRoot() as repo_path:
            sys.path.insert(0, os.path.join(repo_path, "src", "lib"))
            # Removed cuda ops dependencies
            find_replace_in_repo(
                repo_path,
                "src/lib/models/networks/DCNv2/dcn_v2_func.py",
                "from ._ext import dcn_v2 as _backend",
                "# from ._ext import dcn_v2 as _backend",
            )

            from models.networks.DCNv2.dcn_v2 import DCN
            from models.networks.pose_dla_dcn import get_pose_net

            DCN.forward = custom_dcn_forward
            model = get_pose_net(
                num_layers=34,
                heads=heads,
                head_conv=256,
            )
            model = cast(CenterNet, load_model(model, ckpt_path))
            model.eval()
        return model

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device=None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True"

        return compile_options

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "image": (
                (batch_size, 3, height, width),
                "float32",
            ),
        }


def load_model(model: nn.Module, model_path: str) -> nn.Module:
    checkpoint = torch.load(model_path)
    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model.load_state_dict(state_dict, strict=False)
    return model
