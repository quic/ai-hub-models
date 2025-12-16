# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.layers import ROIAlign
from detectron2.model_zoo import get_checkpoint_url, get_config_file
from detectron2.modeling import build_model

from qai_hub_models.models._shared.detectron2.model_patches import ROIAlign_forward
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
)
from qai_hub_models.utils.base_model import BaseModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image.jpg"
)


class Detectron2(BaseModel):
    @classmethod
    def from_pretrained(cls, config: str):
        # Make Functional in QNN
        ROIAlign.forward = ROIAlign_forward

        cfg = get_cfg()
        cfg.merge_from_file(get_config_file(config))
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()

        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)

        weight_name = Path(config).stem
        weight = str(
            CachedWebModelAsset(
                get_checkpoint_url(config),
                MODEL_ID,
                MODEL_ASSET_VERSION,
                f"{weight_name}.pkl",
            ).fetch()
        )
        checkpointer.load(weight)
        return cls(model)

    @staticmethod
    def calibration_dataset_name() -> str:
        return "coco"
