# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.models._shared.deeplab.model import DeepLabV3Model
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
    load_torch,
)
from qai_hub_models.utils.input_spec import InputSpec

SOURCE_REPO = "https://github.com/LikeLy-Journey/SegmenTron"
COMMIT_HASH = "4bc605eedde7d680314f63d329277b73f83b1c5f"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/deeplabv3_plus_xception_pascal_voc_segmentron.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "deeplabv3_plus_xception_pascal_voc_segmentron.pth",
)


class DeeplabXception(DeepLabV3Model):
    def __init__(self, deeplabv3_model: torch.nn.Module, normalize_input: bool = False):
        super().__init__(deeplabv3_model, normalize_input=normalize_input)

    @classmethod
    def from_pretrained(cls, weights_url: str | CachedWebModelAsset = DEFAULT_WEIGHTS):
        weights = load_torch(weights_url)
        with SourceAsRoot(
            SOURCE_REPO, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION
        ) as repo_path:
            find_replace_in_repo(
                repo_path,
                "segmentron/models/__init__.py",
                "from .ccnet import CCNet",
                "",
            )
            from segmentron.config import cfg
            from segmentron.models.model_zoo import get_segmentation_model

            if weights_url == DEFAULT_WEIGHTS:
                config_file = "configs/pascal_voc_deeplabv3_plus.yaml"
            else:
                raise ValueError("Unrecognized weights")

            cfg.update_from_file(config_file)
            model = get_segmentation_model()
            model.load_state_dict(weights)
            instance = cls(model).eval()
            if (
                hasattr(instance.model, "encoder")
                and hasattr(instance.model.encoder, "named_modules")
                and hasattr(cfg.MODEL, "BN_EPS_FOR_ENCODER")
                and cfg.MODEL.BN_EPS_FOR_ENCODER
            ):

                for name, module in instance.model.encoder.named_modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        module.eps = cfg.MODEL.BN_EPS_FOR_ENCODER

            return instance

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run DeepLabV3_Plus_Mobilenet on `image`, and produce a tensor of classes for segmentation

        Parameters:
            image: Pixel values pre-processed for model consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            tensor: Bx21xHxW tensor of class logits per pixel
        """

        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
        normalized_image = (image - mean) / std

        model_out = self.model(normalized_image)
        if isinstance(model_out, tuple):
            model_out = model_out[0]
        return model_out.argmax(1).byte()

    @staticmethod
    def get_input_spec(
        height: int = 480,
        width: int = 520,
    ) -> InputSpec:
        return {"image": ((1, 3, height, width), "float32")}
