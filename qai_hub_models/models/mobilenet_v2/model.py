# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json

import torch

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
MOBILENETV2_WEIGHTS = "mobilenet_v2.pth.tar"
# MOBILENETV2_WEIGHTS = "torch_mobilenetv2_w8a8_state_dict.pth"
# from https://github.com/quic/aimet-model-zoo/blob/d09d2b0404d10f71a7640a87e9d5e5257b028802/aimet_zoo_torch/mobilenetv2/model/model_cards/mobilenetv2_w8a8.json
MOBILENETV2_CFG = "mobilenetv2_w8a8.json"
MOBILENETV2_SOURCE_REPOSITORY = "https://github.com/tonylins/pytorch-mobilenet-v2"
MOBILENETV2_SOURCE_REPO_COMMIT = "99f213657e97de463c11c9e0eaca3bda598e8b3f"


class MobileNetV2(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = MOBILENETV2_WEIGHTS) -> MobileNetV2:
        model = _load_mobilenet_v2_source_model()
        checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, weights
        ).fetch()
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        # rename classifier.1.weight -> classifier.weight, and bias similarly
        state_dict = {
            k.replace("classifier.1", "classifier"): v for k, v in checkpoint.items()
        }
        model.load_state_dict(state_dict)

        return cls(model)


def _load_mobilenet_v2_source_model() -> torch.nn.Module:
    cfg_path = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, MOBILENETV2_CFG
    ).fetch()
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    with SourceAsRoot(
        MOBILENETV2_SOURCE_REPOSITORY,
        MOBILENETV2_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        # necessary import. `modeling.deeplab` comes from the DeepLabV3 repo.
        from MobileNetV2 import MobileNetV2 as _MobileNetV2

        return _MobileNetV2(
            n_class=cfg["model_args"]["num_classes"],
            input_size=cfg["model_args"]["input_size"],
            width_mult=cfg["model_args"]["width_mult"],
        )
