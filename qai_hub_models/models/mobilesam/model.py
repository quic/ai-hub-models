# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.sam.model import SAMDecoder, SAMLoader
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MobileSAM_SOURCE_REPO = "https://github.com/ChaoningZhang/MobileSAM.git"
MobileSAM_SOURCE_REPO_COMMIT = "c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed"

MODEL_ID = __name__.split(".")[-2]
DEFAULT_MODEL_TYPE = "vit_t"
SMALL_MODEL_TYPE = "vit_t"
MODEL_REGISTRY = {
    "vit_t": "mobile_sam.pt",
}
MODEL_ASSET_VERSION = 1
MODEL_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, MODEL_REGISTRY["vit_t"]
)


with SourceAsRoot(
    MobileSAM_SOURCE_REPO,
    MobileSAM_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
) as repo_path:
    from mobile_sam import SamPredictor, sam_model_registry  # noqa: F401
    from mobile_sam.modeling.sam import Sam
    from mobile_sam.utils.onnx import SamOnnxModel  # noqa: F401


class MobileSAMEncoder(BaseModel):
    def __init__(self, sam: Sam):
        super().__init__()
        self.sam = sam

    def forward(self, image: torch.Tensor):
        x = self.sam.preprocess(image)
        return self.sam.image_encoder(x)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        encoder_img_height: int = 1024,  # self.sam.image_encoder.img_size
        encoder_img_width: int = 1024,  # self.sam.image_encoder.img_size
    ) -> InputSpec:
        return {
            "image": ((batch_size, 3, encoder_img_height, encoder_img_width), "float32")
        }

    def _get_input_spec_for_instance(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return self.__class__.get_input_spec(
            batch_size,
            self.sam.image_encoder.img_size,
            self.sam.image_encoder.img_size,
        )

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["image_embeddings"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["image_embeddings"]

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> MobileSAMEncoder:
        return MobileSAMEncoder(MobileSAMLoader._load_sam_from_repo(model_type))


@CollectionModel.add_component(MobileSAMEncoder, "SAMEncoder")
@CollectionModel.add_component(SAMDecoder)
class MobileSAM(CollectionModel):
    def __init__(self, sam: Sam, encoder: MobileSAMEncoder, decoder: SAMDecoder):
        super().__init__(encoder, decoder)
        self.sam = sam
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_pretrained(
        cls, model_type: str = DEFAULT_MODEL_TYPE, single_mask_mode: bool = True
    ) -> MobileSAM:
        return cls(*MobileSAMLoader.load(model_type, single_mask_mode))


class MobileSAMLoader:
    @classmethod
    def load(
        cls,
        model_type: str = DEFAULT_MODEL_TYPE,
        single_mask_mode: bool = True,
    ) -> tuple[Sam, MobileSAMEncoder, SAMDecoder]:
        sam = cls._load_sam_from_repo(model_type)
        cls._patch_sam_for_qnn_comatibility(sam)
        return sam, MobileSAMEncoder(sam), SAMDecoder(sam, single_mask_mode)

    @staticmethod
    def _load_sam_from_repo(model_type: str = DEFAULT_MODEL_TYPE) -> Sam:
        weight_asset = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, MODEL_REGISTRY[model_type]
        )
        weight_asset.fetch()
        return sam_model_registry[model_type](weight_asset.path())

    @staticmethod
    def _patch_sam_for_qnn_comatibility(sam: Sam) -> None:
        return SAMLoader._patch_sam_for_qnn_comatibility(sam, patch_encoder=False)
