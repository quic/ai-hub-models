# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from qai_hub_models.evaluators.kitti_evaluator import BaseEvaluator, KittiEvaluator
from qai_hub_models.models._shared.centernet.model import CenterNet, CenterNetAsRoot
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.image_processing import (
    normalize_image_torchvision,
    pre_process_with_affine,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# demo image was downloaded from kitti dataset
IMAGE = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "image.png")

# checkpoint download from https://drive.google.com/file/d/1LrAzVJqlZECVuyr_NJI_4xd88mA1fL5b
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "ddd_3dop.pth",
)


class CenterNet3D(CenterNet):
    """
    CenterNet 3D Object Detection

    Parameters
    ----------
        model (BaseModule): Centernet 3D bbox model.
        ddd_decode(Callable): 3D bbbox dectection decoder function.
    """

    def __init__(
        self,
        model: nn.Module,
        ddd_decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ],
    ) -> None:
        super().__init__()
        self.model = model
        self.decode = ddd_decode

    @classmethod
    def from_pretrained(cls, ckpt_path: str = "default") -> CenterNet3D:
        heads = {"hm": 3, "dep": 1, "rot": 8, "dim": 3, "wh": 2, "reg": 2}
        if ckpt_path == "default":
            ckpt_path = str(DEFAULT_WEIGHTS.fetch())
        model = super().from_pretrained(ckpt_path, heads)
        with CenterNetAsRoot() as repo_path:
            sys.path.insert(0, os.path.join(repo_path, "src", "lib"))
            from models.decode import ddd_decode
        return cls(model, ddd_decode)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run CenterNet3D model and returns hm, dep, rot, dim, wh, reg.

        Parameters
        ----------
            B = batch size, C = 3, H = img height, W = img width
            imgs: torch.Tensor of shape [B,C,H,W] as float32
                Preprocessed image with range[0-1] in RGB format.

        Returns
        -------
            hm (torch.Tensor): Heatmap with the shape of
                [B, num_classes, H//4, W//4].
            dep (torch.Tensor): depth value with the
                shape of [B, 1, H//4, W//4].
            rot (torch.Tensor): Rotation value with the
                shape of [B, 8, H//4, W//4].
            dim (torch.Tensor): Size value with the shape
                of [B, 3, H//4, W//4].
            wh (torch.Tensor): Width/Height value with the
                shape of [B, 2, H//4, W//4].
            reg (torch.Tensor): 2D regression value with the
                shape of [B, 2, H//4, W//4].
        """
        image = image[:, [2, 1, 0]]
        image = normalize_image_torchvision(image)

        hm, dep, rot, dim, wh, reg = self.model(image)[-1].values()
        hm = torch.sigmoid(hm)
        dep = torch.exp(-dep)

        return hm, dep, rot, dim, wh, reg

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 384,
        width: int = 1280,
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

    @staticmethod
    def get_output_names() -> list[str]:
        return ["hm", "dep", "rot", "dim", "wh", "reg"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = np.array(load_image(IMAGE.fetch()))
        h, w = self.get_input_spec()["image"][0][2:]
        height, width = image.shape[0:2]
        c = np.array([width / 2, height / 2])
        s = np.array([width, height])
        img = pre_process_with_affine(image, c, s, 0, (h, w))
        return {
            "image": [img.numpy()],
        }

    def get_evaluator(self) -> BaseEvaluator:
        return KittiEvaluator(self.decode)

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["kitti"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "kitti"


def load_model(model, model_path):
    checkpoint = torch.load(model_path, weights_only=False)

    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model.load_state_dict(state_dict, strict=False)
    return model
